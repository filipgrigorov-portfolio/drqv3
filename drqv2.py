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

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
from PIL import Image

# TODO: Place in cli args
STAGE_1 = True
STAGE_2 = False

LOG_EVERY = 50000

LAMBDA = 0.1 # reward loss weight
GAMMA = 0.9 # reward_pred vs reward

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
        self.image_shape = (9, 84, 84)
        self.flat_image_shape = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]

        # t-SNE
        self.scaler = StandardScaler()
        self.tsne = TSNE(n_components=2, perplexity=30, random_state=42)  # Adjust perplexity based on dataset size

        # models
        # Encoder
        if STAGE_1:
            # flat shape as we need 9 rgbs + states
            state_shape = obs_shape[0] - self.flat_image_shape
            self.state_encoder = StateEncoder(state_shape).to(device)
            self.image_encoder = ImageEncoder(self.image_shape).to(device)
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
            self.reconstructor = cVAE(input_shape=self.image_shape, latent_dim=128, context_dim=action_shape[0], freeze_encoder=STAGE_2).to(device)
        elif STAGE_2:
            self.reconstructor = cVAE(input_shape=obs_shape, latent_dim=128, context_dim=action_shape[0], freeze_encoder=STAGE_2).to(device)



        # NOTE (informative): optimizers
        self.image_encoder_opt = torch.optim.Adam(self.image_encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        LR_VAE = lr #1e-3
        self.cVAE_opt = torch.optim.Adam(self.reconstructor.parameters(), lr=LR_VAE) # NOTE (informative)



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
        if STAGE_1:
            obs = obs[self.flat_image_shape:]
            obs = self.state_encoder(obs)    # STATES ENCODING 
        elif STAGE_2:
            obs = self.image_encoder(obs.unsqueeze(0)) # IMG ENCODING
        else:
            raise("Just STAGE_1 and STAGE_2")

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev) # ACTOR POLICY
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, reward_pred, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            if reward_pred is not None:
                reward_loss = torch.nn.functional.mse_loss(reward, reward_pred)
            #reward = torch.min(reward, reward_pred) # experiment (lower bound)
            #reward = GAMMA * reward + (1.0 - GAMMA) * reward_pred
            target_Q = reward + (discount * target_V)
        
        Q1, Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)# + LAMBDA * reward_loss

        if self.use_tb:
            if reward_pred is not None:
                metrics['reward_pred'] = reward_pred.mean().item()
            metrics['reward'] = reward.mean().item()
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            if reward_pred is not None:
                metrics['reward_loss'] = reward_loss.item()

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

        actor_loss = -Q.mean() # max the prob(a) to max Q for said action

        # NOTE (informative): optimize actor
        self.actor_opt.zero_grad(set_to_none=True)

        actor_loss.backward()
        
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
    
    def update_reconstruction(self, obs, actions, step, num_steps):
        metrics = dict()

        obs = obs[:, :self.flat_image_shape].view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2]) if STAGE_1 else obs # Extract for stage 1 and use as is for stage 2 (images only)

        obs_reconstructed, mu, logvar, z, reward_pred = self.reconstructor(x=obs, context=actions)
        kl_weight = 1.0 #min(step / int(0.9 * num_steps), 1.0)
        reconstruction_loss, recon_loss, kl_loss = compute_reconstruction_loss(reconstructed=obs_reconstructed, original=obs, mu=mu, logvar=logvar, kl_weight=kl_weight)

        self.cVAE_opt.zero_grad()

        reconstruction_loss.backward()

        self.cVAE_opt.step()

        if self.use_tb:
            variance = torch.exp(0.5 * logvar)
            std_dev = torch.sqrt(variance)

            metrics["cVAE/recon_loss"] = recon_loss.mean().item()
            metrics["cVAE/kl_loss"] = kl_loss.mean().item()
            metrics["cVAE/distribution_mean"] = mu.mean().item()
            metrics["cVAE/distribution_variance"] = variance.mean().item()
            #metrics["cVAE/distribution_stddev"] = std_dev.mean().item()
            metrics["cVAE/reconstruction_loss"] = reconstruction_loss.item()

            if step == 0 or step % LOG_EVERY == 0:
                metrics["cVAE/original_images"] = obs[:3, :3, ...]
                metrics["cVAE/reconstructed_images"] = obs_reconstructed[:3, :3, ...]

                latent_vectors_normalized = self.scaler.fit_transform(z.detach().cpu().numpy())
                latent_2d = self.tsne.fit_transform(latent_vectors_normalized)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], s=10, alpha=0.7)
                plt.title("t-SNE Visualization of cVAE Latent Space")
                plt.xlabel("t-SNE Dimension 1")
                plt.ylabel("t-SNE Dimension 2")

                # Convert plot to a numpy array
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image = Image.open(buf)
                image_array = np.array(image)
                buf.close()
                plt.close()

                metrics["cVAE/latent_space"] = image_array

        return metrics, reward_pred

    def update(self, replay_iter, step, num_steps):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # NOTE (informative): augment -> replace with cVAE in the replay buffer
        #obs = self.aug(obs.float())
        #next_obs = self.aug(next_obs.float())

        # encode
        images, next_images = None, None
        if STAGE_1:
            states = obs[:, self.flat_image_shape:]
            images = obs[:, :self.flat_image_shape]
            obs = self.state_encoder(states)
            image_feats = self.image_encoder(images.view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])) #TODO: What to do with this?
            with torch.no_grad():
                next_states = next_obs[:, self.flat_image_shape:]
                next_images = next_obs[:, :self.flat_image_shape]
                next_obs = self.state_encoder(next_states)
                next_images = self.image_encoder(next_images.view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2]))

        elif STAGE_2:
            images = obs.clone()
            obs = self.image_encoder(obs)
            with torch.no_grad():
                next_images = next_obs.clone()
                next_obs = self.image_encoder(next_obs)
        else:
            raise("Just STAGE_1 and STAGE_2")

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, None, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # NOTE (important): This needs to be after the updates to the above policies to train properly!!!
        # update reconstruction
        if STAGE_1:
            reconstruction_metrics, reward_pred = self.update_reconstruction(images.detach(), action, step, num_steps)
            metrics.update(reconstruction_metrics)
        elif STAGE_2:
            # might not be turned on
            reconstruction_metrics, reward_pred = self.update_reconstruction(images.detach(), action, step, num_steps)
            metrics.update(reconstruction_metrics)
        else:
            raise("Just STAGE_1 and STAGE_2")

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
