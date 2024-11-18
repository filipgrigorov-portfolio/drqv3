# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from models import *

# TODO: Place in cli args
STAGE_1 = True
STAGE_2 = False

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        # Encoder
        if STAGE_1:
            # flat shape as we need 9 rgbs + states
            image_shape = (9, 84, 84)
            flat_image_shape = image_shape[0] * image_shape[1] * image_shape[2]
            state_shape = (obs_shape - flat_image_shape)
            self.image_encoder = StateEncoder(state_shape).to(device)
            self.image_encoder = ImageEncoder(image_shape).to(device)
        elif STAGE_2:
            # image only with the right shape (not flat)
            self.image_encoder = ImageEncoder(obs_shape).to(device)

        # Actor policy
        self.actor = Actor(self.image_encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)

        # Critic policy
        self.critic = Critic(self.image_encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        
        # NOTE: These are updated every now and then via EMA (2 target critic networks)
        self.critic_target = Critic(self.image_encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())


        # NOTE: cVAE framework (conditioned on action_shape)
        if STAGE_1:
            self.reconstructor = cVAE(input_shape=image_shape, latent_dim=128, context_dim=action_shape, freeze_encoder=STAGE_2)
        elif STAGE_2:
            self.reconstructor = cVAE(input_shape=obs_shape, latent_dim=128, context_dim=action_shape, freeze_encoder=STAGE_2)



        # NOTE (informative): optimizers
        self.image_encoder_opt = torch.optim.Adam(self.image_encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.cVAE_opt = torch.optim.Adam(self.reconstructor.parameters(), lr=lr) # NOTE (informative)



        # TODO: to be replaced by the states from cVAE when training on visual input
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)



        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.image_encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.image_encoder(obs.unsqueeze(0)) # IMG ENCODING -> STATES ENCODING 
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev) # ACTOR POLICY
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # NOTE (informative): optimize encoder and critic
        self.image_encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward()
        
        self.critic_opt.step()
        self.image_encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # NOTE (informative): optimize actor
        self.actor_opt.zero_grad(set_to_none=True)

        actor_loss.backward()
        
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
    
    def update_reconstruction(self, obs, step):
        metrics = dict()

        obs_reconstructed, mu, logvar = self.reconstructor(x=obs)
        reconstruction_loss = compute_reconstruction_loss(reconstructed=obs_reconstructed, original=obs)

        self.cVAE_opt.zero_grad()

        reconstruction_loss.backward()

        self.cVAE_opt.step()

        if self.use_tb:
            variance = torch.exp(logvar)
            std_dev = torch.sqrt(variance)

            metrics("Distribution/Mean", mu.mean().item())
            metrics("Distribution/Variance", variance.mean().item())
            metrics("Distribution/StdDev", std_dev.mean().item())
            metrics['reconstruction_loss'] = reconstruction_loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # NOTE (informative): augment -> replace with cVAE in the replay buffer
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        # encode
        obs = self.image_encoder(obs)
        with torch.no_grad():
            next_obs = self.image_encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        
        # update reconstruction
        if STAGE_1:
            metrics.update(self.update_reconstruction(obs.detach(), step))

        return metrics
