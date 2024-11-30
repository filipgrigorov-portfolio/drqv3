# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

import utils

from models import *
from losses import *

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
from PIL import Image

# TODO: Place in cli args
STAGE_1 = True
STAGE_2 = False

LOG_EVERY = 8000
NUM_ACC_STEPS = 1000
UPDATE_DISCRIMINATOR_EVERY = 300

REWARD_LOSS_WEIGHT = 0.3 # reward loss weight
GAMMA = 0.7 # reward_pred vs reward (NOT quite working)
PLOT_LATENT = False

WITH_MASKS = False
WITH_DYNAMIC_MASK_ONLY = False

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
            self.state_encoder.train()

        LATENT_DIM = 128
        self.reconstructor = cVAE(
            input_shape=self.image_shape, 
            latent_dim=LATENT_DIM, 
            context_actions_dim=action_shape[0], 
            context_rewards_dim=1, 
            freeze_encoder=STAGE_2).to(device)
        self.reconstructor.train()

        # Actor policy
        self.actor = Actor(self.reconstructor.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)

        self.prev_actor_loss = 1e10
        self.prev_reconstruction_loss = 1e10
        if STAGE_2:
            print(f"Loading pretrained actor weights at actor_weights.pth")
            self.pretrained_actor = Actor(self.reconstructor.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
            self.pretrained_actor.load_state_dict(torch.load("actor_weights.pth", weights_only=True))
            self.pretrained_actor.eval()

            print(f"Loading pretrained state encoder weights at state_encoder_weights.pth")
            self.state_encoder = StateEncoder(state_shape).to(device)
            self.state_encoder.load_state_dict(torch.load("state_encoder_weights.pth", weights_only=True))
            self.state_encoder.eval()

            print(f"Loading pretrained cVAE weights at cVAE_weights.pth")
            self.reconstructor.load_state_dict(torch.load("cVAE_weights.pth", weights_only=True))
            #self.reconstructor.eval() # This could train some more or not
            

        # Critic policy
        self.critic = Critic(self.reconstructor.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        
        # NOTE: These are updated every now and then via EMA (2 target critic networks)
        self.critic_target = Critic(self.reconstructor.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        #self.reward_pred = None

        self.ssim_loss_fn = SSIMLoss(window_size=11, channel=3, reduction='mean').to(device)

        # NOTE (informative): optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam([
            {"params": self.critic.parameters(), "lr": lr},
            {"params": self.state_encoder.parameters(), "lr": lr}
        ])
        LR_VAE = lr#1e-3
        self.cVAE_opt = torch.optim.Adam(self.reconstructor.parameters(), lr=LR_VAE) # NOTE (informative)

        # NOTE (experiment):
        self.discriminator_latent = LatentDiscriminator(latent_dim=LATENT_DIM).to(device)
        self.discriminator_image = ImageDiscriminator(input_channels=self.reconstructor.input_shape[0]).to(device)
        torch.nn.utils.clip_grad_norm_(self.discriminator_latent.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.discriminator_image.parameters(), max_norm=1.0)
        self.optim_discriminator = torch.optim.Adam(list(self.discriminator_latent.parameters()) + list(self.discriminator_image.parameters()), lr=1e-5)


        # TODO: to be replaced by the states from cVAE when training on visual input
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        if STAGE_1:
            obs = obs[self.flat_image_shape:]
            obs = self.state_encoder(obs)    # STATES ENCODING 
        elif STAGE_2:
            obs = self.reconstructor.encode(obs.unsqueeze(0)) # IMG ENCODING
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
            metrics['reward'] = reward.mean().item()
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # NOTE (informative): optimize encoder and critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        return metrics

    # TODO: Add regularization (KL) from pretrained actor policy, from STAGE_1
    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(obs, stddev)
        
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean() # max the prob(a) to max Q for said action

        if STAGE_2:
            actor_kl_reg = compute_kl_divergence_on_behaviour_policy(self.actor, self.pretrained_actor, obs)
            if self.use_tb:
                metrics['actor_kl_reg'] = actor_kl_reg.item()
            actor_loss = actor_loss + 0.1 * actor_kl_reg

        # NOTE (informative): optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # NOTE (Save for stage 2)
        if actor_loss.item() < self.prev_actor_loss:
            print(f"Saving actor weights at actor_weights.pth")
            torch.save(self.actor.state_dict(), "actor_weights.pth")

            print(f"Saving actor weights at state_encoder_weights.pth")
            torch.save(self.state_encoder.state_dict(), "state_encoder_weights.pth")
            
            self.prev_actor_loss = actor_loss.item()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_discrimination(self, obs, actions, rewards, step, num_steps):
        """Train discriminator to classify between real and fake data"""
        metrics = dict()

        self.reconstructor.requires_grad_(False)
        self.discriminator_latent.requires_grad_(True)
        self.discriminator_image.requires_grad_(True)

        obs_real = obs[:, :self.flat_image_shape].view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2]) if STAGE_1 else obs # Extract for stage 1 and use as is for stage 2 (images only)
        obs_fake, _, _, z_fake = self.reconstructor(x=obs_real, context_actions=actions, context_rewards=rewards)
        #z_real = torch.normal(mean=0, std=1, size=(z_fake.shape[0], z_fake.shape[1]), device=z_fake.device)

        #latent_real = self.discriminator_latent(z_real)
        #latent_fake = self.discriminator_latent(z_fake.detach())
        #d_loss_latent = -torch.mean(torch.log(latent_real) + torch.log(1.0 - latent_fake))

        img_real = self.discriminator_image(obs_real)
        img_fake = self.discriminator_image(obs_fake.detach())
        d_loss_recon = -torch.mean(torch.log(img_real) + torch.log(1.0 - img_fake))

        #d_loss = d_loss_recon

        #metrics["discriminator/d_loss_latent"] = d_loss_latent.mean().item()
        metrics["discriminator/d_loss_recon"] = d_loss_recon.mean().item()
        #metrics["discriminator/d_loss"] = d_loss.mean().item()

        if step % UPDATE_DISCRIMINATOR_EVERY == 0:
            self.optim_discriminator.zero_grad()
            d_loss_recon.backward()
            self.optim_discriminator.step()

        return metrics

    
    def update_reconstruction(self, obs, actions, rewards, step, num_steps):
        """Train the cVAE to reconstruct the input images"""
        metrics = dict()

        # NOTE (experiment):
        # Turn off backprop for the discriminator (use as fixed loss generator, "critic" of the input)
        self.reconstructor.requires_grad_(True)
        self.discriminator_latent.requires_grad_(False)
        self.discriminator_image.requires_grad_(False)



        obs = obs[:, :self.flat_image_shape].view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2]) if STAGE_1 else obs # Extract for stage 1 and use as is for stage 2 (images only)

        obs_dynamic = None
        if WITH_MASKS:
            # Mask in dynamic parts
            frame_diff = compute_frame_difference(obs, num_frames=obs.shape[1] // 3)
            mask = create_dynamic_mask(frame_diff=frame_diff, threshold=0.1, mode="dynamic")
            obs_dynamic = mask_inputs_with_dynamic_regions(obs, mask, num_frames=obs.shape[1] // 3, mode="dynamic")
            # ---
            
            # Mask in static parts
            frame_diff = compute_frame_difference(obs, num_frames=obs.shape[1] // 3)
            mask = create_dynamic_mask(frame_diff=frame_diff, threshold=0.1, mode="static")
            obs_static = mask_inputs_with_dynamic_regions(obs, mask, num_frames=obs.shape[1] // 3, mode="static")
            # ---

            #action_normalized = actions / actions.norm(dim=-1, keepdim=True)
            obs_dynamic_reconstructed, mu_dynamic, logvar_dynamic, _, _ = self.reconstructor(x=obs_dynamic, context_actions=actions, context_rewards=rewards)
            #obs_static_reconstructed, mu, logvar, z, reward_pred = self.reconstructor(x=obs_static, context=action_normalized)
            obs_static_reconstructed, mu, logvar, z = self.reconstructor(x=obs_static, context_actions=actions, context_rewards=rewards)

            kl_weight = min(step / int(0.7 * num_steps), 1.0)
            reconstruction_loss_dynamic, _, _ = compute_reconstruction_loss(reconstructed=obs_dynamic_reconstructed, original=obs, mu=mu_dynamic, logvar=logvar_dynamic, kl_weight=kl_weight)
            reconstruction_loss_static, recon_loss, kl_loss = compute_reconstruction_loss(reconstructed=obs_static_reconstructed, original=obs, mu=mu, logvar=logvar, kl_weight=kl_weight)

        elif WITH_DYNAMIC_MASK_ONLY:
            frame_diff = compute_frame_difference(obs, num_frames=obs.shape[1] // 3)
            mask = create_dynamic_mask(frame_diff=frame_diff, threshold=0.1, mode="dynamic")
            obs_dynamic = mask_inputs_with_dynamic_regions(obs, mask, num_frames=obs.shape[1] // 3, mode="dynamic")
            obs_reconstructed, mu, logvar, z = self.reconstructor(x=obs_dynamic, context_actions=actions, context_rewards=rewards)

            kl_weight = min(step / int(0.7 * num_steps), 1.0)
            reconstruction_loss, recon_loss, kl_loss = compute_reconstruction_loss(reconstructed=obs_reconstructed, original=obs_dynamic, mu=mu, logvar=logvar, kl_weight=kl_weight)

        else:
            #action_normalized = actions / actions.norm(dim=-1, keepdim=True)
            #obs_reconstructed, mu, logvar, z, reward_pred = self.reconstructor(x=obs, context=action_normalized)
            obs_reconstructed, mu, logvar, z = self.reconstructor(x=obs, context_actions=actions, context_rewards=rewards)

            kl_weight = min(step / int(0.7 * num_steps), 1.0)
            reconstruction_loss, recon_loss, kl_loss = compute_reconstruction_loss(reconstructed=obs_reconstructed, original=obs, mu=mu, logvar=logvar, kl_weight=kl_weight)


        
        # SSIM loss
        ssim_loss = 0.0
        for i in range(0, obs.shape[1] // 3):
            if WITH_MASKS:
                ssim_loss += self.ssim_loss_fn(obs_dynamic_reconstructed[:, i*3:(i+1)*3, ...], obs[:, i*3:(i+1)*3, ...])
                ssim_loss += self.ssim_loss_fn(obs_static_reconstructed[:, i*3:(i+1)*3, ...], obs[:, i*3:(i+1)*3, ...])
            elif WITH_DYNAMIC_MASK_ONLY:
                ssim_loss += self.ssim_loss_fn(obs_reconstructed[:, i*3:(i+1)*3, ...], obs_dynamic[:, i*3:(i+1)*3, ...])
            else:
                ssim_loss += self.ssim_loss_fn(obs_reconstructed[:, i*3:(i+1)*3, ...], obs[:, i*3:(i+1)*3, ...])

        if WITH_MASKS:
            ssim_loss /= (obs.shape[1] * 2)
        else:
            ssim_loss /= obs.shape[1]



        # Reward loss (from reward head)
        # def standardize(x):
        #     """Good for values having +ve and -ve values to normalize in [0, 1]"""
        #     return (x - x.mean()) / (x.std() + 1e-8)
        # reward_normalized = standardize(rewards)
        # reward_pred_normalized = standardize(reward_pred)
        # print(f"{reward_normalized.mean()} += {reward_normalized.std()}, {reward_normalized.min()}, {reward_normalized.max()}")
        # print(f"{reward_pred_normalized.mean()} += {reward_pred_normalized.std()}, {reward_pred_normalized.min()}, {reward_pred_normalized.max()}")
        # reward_loss = F.mse_loss(reward_normalized, reward_pred_normalized, reduction="mean")


        # NOTE (experiment):
        adv_image_loss = -torch.mean(torch.log(self.discriminator_image(obs_reconstructed)).detach())
        #adv_latent_loss = -torch.mean(torch.log(self.discriminator_latent(z)))
        metrics["cVAE/adv_image_loss"] = adv_image_loss.mean().item()
        #metrics["cVAE/adv_latent_loss"] = adv_latent_loss.mean().item()


        if WITH_MASKS:
            total_loss = reconstruction_loss_static + reconstruction_loss_dynamic + ssim_loss# + REWARD_LOSS_WEIGHT * reward_loss
        else:
            total_loss = reconstruction_loss + 0.3 * ssim_loss + 0.1 * adv_image_loss# + REWARD_LOSS_WEIGHT * reward_loss



        self.cVAE_opt.zero_grad()
        total_loss.backward()
        self.cVAE_opt.step()


        if total_loss < self.prev_reconstruction_loss:
            print(f"Saving cVAE weights at cVAE_weights.pth")
            torch.save(self.reconstructor.state_dict(), "cVAE_weights.pth")
            self.prev_reconstruction_loss = total_loss.item()


        if self.use_tb:
            variance = torch.exp(0.5 * logvar)
            #std_dev = torch.sqrt(variance)

            metrics["cVAE/recon_loss"] = recon_loss.mean().item()
            metrics["cVAE/kl_loss"] = kl_loss.mean().item()
            metrics["cVAE/ssim_loss"] = ssim_loss.mean().item()
            #metrics["cVAE/reward_loss"] = reward_loss.mean().item()

            #metrics["cVAE/reward_gt"] = rewards.max().item()
            #metrics["cVAE/reward_pred"] = reward_pred.max().item()

            if WITH_MASKS:
                metrics["cVAE/reconstruction_dynamic_loss"] = reconstruction_loss_dynamic.mean().item()
                metrics["cVAE/reconstruction_static_loss"] = reconstruction_loss_static.mean().item()
            else:
                metrics["cVAE/reconstruction_loss"] = reconstruction_loss.mean().item()

            metrics["cVAE/distribution_mean"] = mu.mean().item()
            metrics["cVAE/distribution_variance"] = variance.mean().item()
            #metrics["cVAE/distribution_stddev"] = std_dev.mean().item()
            metrics["cVAE/total_loss"] = total_loss.item()

            if step == 2000 or step % LOG_EVERY == 0:
                if WITH_MASKS:
                    metrics["cVAE/original_dynamic_images"] = obs_dynamic[:3, :3, ...]
                    metrics["cVAE/original_static_images"] = obs_static[:3, :3, ...]
                    metrics["cVAE/reconstructed_dynamic_images"] = obs_dynamic_reconstructed[:3, :3, ...]
                    metrics["cVAE/reconstructed_static_images"] = obs_static_reconstructed[:3, :3, ...]
                elif WITH_DYNAMIC_MASK_ONLY:
                    metrics["cVAE/original_images1"] = obs_dynamic[:3, :3, ...] # experimental
                    metrics["cVAE/reconstructed_images1"] = obs_reconstructed[:3, :3, ...] # experimental

                    metrics["cVAE/original_images2"] = obs_dynamic[:3, 3:6, ...] # experimental
                    metrics["cVAE/reconstructed_images2"] = obs_reconstructed[:3, 3:6, ...] # experimental

                    metrics["cVAE/original_images3"] = obs_dynamic[:3, 6:9, ...] # experimental
                    metrics["cVAE/reconstructed_images3"] = obs_reconstructed[:3, 6:9, ...] # experimental
                else:
                    metrics["cVAE/original_images1"] = obs[:3, :3, ...] # experimental
                    metrics["cVAE/reconstructed_images1"] = obs_reconstructed[:3, :3, ...] # experimental

                    metrics["cVAE/original_images2"] = obs[:3, 3:6, ...] # experimental
                    metrics["cVAE/reconstructed_images2"] = obs_reconstructed[:3, 3:6, ...] # experimental

                    metrics["cVAE/original_images3"] = obs[:3, 6:9, ...] # experimental
                    metrics["cVAE/reconstructed_images3"] = obs_reconstructed[:3, 6:9, ...] # experimental

                    #grid_images = make_grid(obs_reconstructed[:20, :3, ...], nrow=8, padding=2)
                    #save_image(grid_images, "TEST_IMG.jpg")



                if PLOT_LATENT:
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

        return metrics#, reward_pred

    def update(self, replay_iter, step, num_steps):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # NOTE (informative): augment -> replace with cVAE in the replay buffer
        #obs = self.aug(obs.float())
        #next_obs = self.aug(next_obs.float())

        # encode
        images, next_images = None, None
        if STAGE_1:
            states = obs[:, self.flat_image_shape:]
            images = obs[:, :self.flat_image_shape]
            obs = self.state_encoder(states)
            with torch.no_grad():
                next_states = next_obs[:, self.flat_image_shape:]
                next_images = next_obs[:, :self.flat_image_shape]
                next_obs = self.state_encoder(next_states)

        # NOTE: This serves as the sampled augmented entries to update the policies instead of detreminitically augmenting the observations --- Address
        elif STAGE_2:
            # NOTE: Load the cVAE weights and the actor policy weights
            images = obs.clone()
            obs = self.reconstructor.encode(obs)
            with torch.no_grad():
                next_images = next_obs.clone()
                next_obs = self.reconstructor.encode(next_images)

            actions_norm = action / action.norm(dim=-1, keepdim=True)
            reward_norm = reward / reward.norm(dim=-1, keepdim=True)
            sampled_obs = self.reconstructor.sample(context_actions=actions_norm, context_rewards=reward_norm)
            # The actions and rewards are more or less similar for immediate next state
            sampled_next_obs = self.reconstructor.sample(context_actions=actions_norm, context_rewards=reward_norm)
            obs = torch.cat([obs, sampled_obs], dim=0)
            next_obs = torch.cat([next_obs, sampled_next_obs], dim=0)
            # TODO: Populate the replay buffer with these, I can replace some of the original entries
        else:
            raise("Just STAGE_1 and STAGE_2")

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        

        # update critic
        metrics_critic = self.update_critic(obs, action, reward, discount, next_obs, step)
        metrics.update(metrics_critic)

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        



        # NOTE (important): This needs to be after the updates to the above policies to train properly!!!
        # update reconstruction
        if STAGE_1:
            # try to select the top reward images
            best_indices = torch.argsort(reward, descending=True, dim=0)[:100]
            high_reward_images = images[best_indices].squeeze(1)
            #sampled_obs = torch.cat([next_images, high_reward_images], dim=0)
            high_rewards = reward[best_indices].squeeze(1)
            high_actions = action[best_indices].squeeze(1)

            # debug
            #print(f"{high_rewards.mean()}, {high_rewards.min()}, {high_rewards.max()}")
            #print(high_rewards.flatten())
            # debug
            
            #self.cVAE_opt.zero_grad()
            ###NUM_ITERS = 3
            ###for _ in range(NUM_ITERS):
            ## actions_norm = action / action.norm(dim=-1, keepdim=True)
            ## reward_norm = reward / reward.norm(dim=-1, keepdim=True)
            ## images = images
            
            actions_norm = high_actions / high_actions.norm(dim=-1, keepdim=True)
            reward_norm = high_rewards / high_rewards.norm(dim=-1, keepdim=True)
            images = high_reward_images

            #TODO: augment???
            images_aug = self.aug(images[:, :self.flat_image_shape].view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2]).float())
            images_aug = images_aug.view(-1, self.flat_image_shape)
            images = torch.cat([images, images_aug], dim=0)
            actions_norm = torch.cat([actions_norm, actions_norm], dim=0)
            reward_norm = torch.cat([reward_norm, reward_norm], dim=0)

            #TODO: shuffle???
            # shuffled_indices = torch.randperm(actions_norm.size(0))
            # actions_norm = actions_norm[shuffled_indices]
            # reward_norm = reward_norm[shuffled_indices]
            # images = images[shuffled_indices]

            # NOTE (experiment): Discriminator step
            discriminator_metrics = self.update_discrimination(obs=images, actions=actions_norm, rewards=reward_norm, step=step, num_steps=num_steps)
            metrics.update(discriminator_metrics)
            
            # NOTE (experiment): Generator step
            reconstruction_metrics = self.update_reconstruction(obs=images, actions=actions_norm, rewards=reward_norm, step=step, num_steps=num_steps)
            metrics.update(reconstruction_metrics)

                ###batch = next(replay_iter)
                ###obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
                ###images = obs[:, :self.flat_image_shape]

            ### self.cVAE_opt.step()

        elif STAGE_2:
            # might not be turned on
            reconstruction_metrics = self.update_reconstruction(images, action, reward, step, num_steps)
            metrics.update(reconstruction_metrics)

        else:
            raise("Just STAGE_1 and STAGE_2")




        return metrics
