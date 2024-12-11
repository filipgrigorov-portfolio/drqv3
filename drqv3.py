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

LOG_EVERY = 200000
NUM_ACC_STEPS = 1000
UPDATE_DISCRIMINATOR_EVERY = 300

REWARD_LOSS_WEIGHT = 0.3 # reward loss weight
GAMMA = 0.7 # reward_pred vs reward (NOT quite working)
PLOT_LATENT = False

WITH_MASKS = False
WITH_DYNAMIC_MASK_ONLY = False

# Training regime flags
# Stage 1
HIGH_REWARD = False
AUGMENT = False
LATENT_DYNAMICS = False
WITH_RECONSTRUCTION = True

# Stage 2
TRAIN_STAGE_2 = True
REGULARIZE_ACTOR = True
SHUFFLE = False

print(f"Training flags: STAGE_1={STAGE_1}, STAGE_2={STAGE_2}, HIGH_REWARD={HIGH_REWARD}, AUGMENT={AUGMENT}, TRAIN_STAGE_2={TRAIN_STAGE_2}, LATENT_DYNAMICS={LATENT_DYNAMICS}, REGULARIZE_ACTOR={REGULARIZE_ACTOR}, SHUFFLE={SHUFFLE}, WITH_RECONSTRUCTION={WITH_RECONSTRUCTION}, WITH_MASKS={WITH_MASKS}, WITH_RECONSTRUCTION={WITH_RECONSTRUCTION}")

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
        return F.grid_sample(x, grid,padding_mode='zeros', align_corners=False)

class NoiseAugmentation(nn.Module):
    def __init__(self, noise_type='gaussian', noise_level=0.1):
        super(NoiseAugmentation, self).__init__()

        self.noise_type = noise_type
        self.noise_level = noise_level

    def forward(self, x):
        if self.noise_type == 'gaussian':
            # Add Gaussian noise: mean 0, std defined by noise_level
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        
        elif self.noise_type == 'uniform':
            # Add uniform noise between [-noise_level, noise_level]
            noise = (torch.rand_like(x) * 2 * self.noise_level) - self.noise_level
            return x + noise
        
        elif self.noise_type == 'salt_and_pepper':
            # Add salt-and-pepper noise: randomly set some pixels to 0 or 1
            mask = torch.rand_like(x) < self.noise_level
            x[mask] = torch.randint(0, 2, size=x[mask].shape, dtype=torch.float)
            return x
        
        else:
            raise ValueError("Unsupported noise type")


class DrQV3Agent:
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
        # flat shape as we need 9 rgbs + states
        state_shape = obs_shape[0] - self.flat_image_shape

        if STAGE_1:
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
            self.image_encoder = ImageEncoder(obs_shape=self.image_shape).to(device)
            self.image_encoder.train()
            self.image_encoder.apply(utils.weight_init)

            print(f"Loading pretrained actor weights at actor_weights.pth")
            self.pretrained_actor = Actor(self.reconstructor.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
            self.pretrained_actor.load_state_dict(torch.load("/home/filip/workspace/repos/research/drqv2/actor_weights.pth", weights_only=True))
            self.pretrained_actor.eval()

            utils.freeze_model(self.pretrained_actor)

            if REGULARIZE_ACTOR:
                print(f"Loading pretrained state encoder weights at state_encoder_weights.pth")
                self.state_encoder = StateEncoder(state_shape).to(device)
                self.state_encoder.load_state_dict(torch.load("/home/filip/workspace/repos/research/drqv2/state_encoder_weights.pth", weights_only=True))
                self.state_encoder.eval()
                utils.freeze_model(self.state_encoder)

            print(f"Loading pretrained cVAE weights at cVAE_weights.pth")
            self.reconstructor.load_state_dict(torch.load("//home/filip/workspace/repos/research/drqv2/cVAE_weights.pth", weights_only=True))
            #self.reconstructor.eval() # This could train some more or not
            # NOTE: We freeze the encoder of the cVAE and, possibly, fine-tune the decoder only
            utils.freeze_model(self.reconstructor.encoder)
            

        # Critic policy
        self.critic = Critic(self.reconstructor.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        
        # NOTE: These are updated every now and then via EMA (2 target critic networks)
        self.critic_target = Critic(self.reconstructor.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.ssim_loss_fn = SSIMLoss(window_size=11, channel=3, reduction='mean').to(device)
        #self.perception_loss = PerceptualLoss().to(device)

        # NOTE (informative): optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        if STAGE_1:
            print(f"Optimizing the state encoder along with the critic in STAGE 1")
            self.critic_opt = torch.optim.Adam([
                {"params": self.critic.parameters(), "lr": lr},
                {"params": self.state_encoder.parameters(), "lr": lr}
            ])
        else:
            print(f"Optimizing the image encoder along with the critic in STAGE 2")
            self.critic_opt = torch.optim.Adam([
                {"params": self.critic.parameters(), "lr": lr},
                {"params": self.image_encoder.parameters(), "lr": lr}
            ])

        LR_VAE = lr #1e-3
        self.cVAE_opt = torch.optim.Adam(self.reconstructor.parameters(), lr=LR_VAE) # NOTE (informative)

        if STAGE_1 and LATENT_DYNAMICS:
           self.latent_dynamics = LatentDynamicsModel(latent_dim=LATENT_DIM, action_dim=action_shape[0]).to(device)

        # NOTE (experiment):
        # self.discriminator_latent = LatentDiscriminator(latent_dim=LATENT_DIM).to(device)
        # self.discriminator_image = ImageDiscriminator(input_channels=self.reconstructor.input_shape[0]).to(device)
        # torch.nn.utils.clip_grad_norm_(self.discriminator_latent.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(self.discriminator_image.parameters(), max_norm=1.0)
        # self.optim_discriminator = torch.optim.Adam(list(self.discriminator_latent.parameters()) + list(self.discriminator_image.parameters()), lr=1e-5)


        # TODO: to be replaced by the states from cVAE when training on visual input
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.aug_noise = NoiseAugmentation(noise_level=0.08)

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
            obs = obs[:self.flat_image_shape].view(self.image_shape[0], self.image_shape[1], self.image_shape[2])
            #obs = self.reconstructor.encode(obs.unsqueeze(0), flatten=True) # IMG ENCODING
            obs = self.image_encoder(obs.unsqueeze(0), flatten=True) # IMG ENCODING
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
    def update_actor(self, obs, step, states_feats=None):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(obs, stddev)
        
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean() # max the prob(a) to max Q for said action

        if STAGE_2 and REGULARIZE_ACTOR:
            B = obs.size(0) // 2
            actor_kl_reg = compute_kl_divergence_on_behaviour_policy(self.actor, self.pretrained_actor, images_feats=obs, states_feats=states_feats, std=stddev)#images_feats=obs[:B, ...], states_feats=states_feats, std=stddev)
            if self.use_tb:
                metrics['actor_kl_reg'] = actor_kl_reg.item()
            actor_loss = actor_loss + GAMMA * actor_kl_reg # TRYING ------

        # NOTE (informative): optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # NOTE (Save for stage 2)
        # if actor_loss.item() < self.prev_actor_loss:
        #     #print(f"Saving actor weights at actor_weights.pth")
        #     torch.save(self.actor.state_dict(), f"actor_weights.pth")

        #     if STAGE_1:
        #         #print(f"Saving actor weights at state_encoder_weights.pth")
        #         torch.save(self.state_encoder.state_dict(), f"state_encoder_weights.pth")
            
        #    self.prev_actor_loss = actor_loss.item()

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

    
    def update_reconstruction(self, obs, next_obs, actions, rewards, step, num_steps):
        """Train the cVAE to reconstruct the input images"""
        metrics = dict()

        # NOTE (experiment):
        # Turn off backprop for the discriminator (use as fixed loss generator, "critic" of the input)
        # self.reconstructor.requires_grad_(True)
        # self.discriminator_latent.requires_grad_(False)
        # self.discriminator_image.requires_grad_(False)



        obs = obs[:, :self.flat_image_shape].view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        next_obs = next_obs[:, :self.flat_image_shape].view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])



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
            obs_reconstructed, mu, logvar, z = self.reconstructor(x=obs, context_actions=actions, context_rewards=rewards)

            if STAGE_1 and LATENT_DYNAMICS:
                _, _, _, z_next = self.reconstructor(x=next_obs, context_actions=actions, context_rewards=rewards)

                # NOTE: compute latent dynamics loss
                z_next_pred = self.latent_dynamics(z, actions)
                latent_dynamics_loss = F.mse_loss(z_next, z_next_pred)

            kl_weight = min(step / num_steps, 1.0)
            reconstruction_loss, recon_loss, kl_loss = compute_reconstruction_loss(reconstructed=obs_reconstructed, original=obs, mu=mu, logvar=logvar, kl_weight=kl_weight)

            #reconstruction_loss_dynamic, _, _ = compute_reconstruction_loss(reconstructed=obs_next_reconstructed, original=next_obs, mu=mu_next, logvar=logvar_next, kl_weight=kl_weight)


        
        # SSIM loss
        ssim_loss = 0.0
        #perceptual_loss = 0.0
        for i in range(0, obs.shape[1] // 3):
            if WITH_MASKS:
                ssim_loss += self.ssim_loss_fn(obs_dynamic_reconstructed[:, i*3:(i+1)*3, ...], obs[:, i*3:(i+1)*3, ...])
                ssim_loss += self.ssim_loss_fn(obs_static_reconstructed[:, i*3:(i+1)*3, ...], obs[:, i*3:(i+1)*3, ...])
            elif WITH_DYNAMIC_MASK_ONLY:
                ssim_loss += self.ssim_loss_fn(obs_reconstructed[:, i*3:(i+1)*3, ...], obs_dynamic[:, i*3:(i+1)*3, ...])
            else:
                ssim_loss += self.ssim_loss_fn(obs_reconstructed[:, i*3:(i+1)*3, ...], obs[:, i*3:(i+1)*3, ...])
                #with torch.no_grad():
                #    perceptual_loss += self.perception_loss(obs_reconstructed[:, i*3:(i+1)*3, ...], obs[:, i*3:(i+1)*3, ...])

        if WITH_MASKS:
            ssim_loss /= (obs.shape[1] * 2)
        else:
            ssim_loss /= obs.shape[1]
            #perceptual_loss /= obs.shape[1]


        # NOTE (experiment):
        #adv_image_loss = -torch.mean(torch.log(self.discriminator_image(obs_reconstructed)).detach())
        #metrics["cVAE/adv_image_loss"] = adv_image_loss.mean().item()


        if WITH_MASKS:
            total_loss = reconstruction_loss_static + reconstruction_loss_dynamic + ssim_loss
        else:
            if STAGE_1:
                total_loss = 0.1 * reconstruction_loss + ssim_loss + 0.5 * (latent_dynamics_loss if LATENT_DYNAMICS else 0.0) # + 0.08 * adv_image_loss
            else:
                total_loss = 0.1 * reconstruction_loss + ssim_loss



        self.cVAE_opt.zero_grad()
        total_loss.backward()
        self.cVAE_opt.step()


        #if total_loss < self.prev_reconstruction_loss:
            #print(f"Saving cVAE weights at cVAE_weights.pth")
            #torch.save(self.reconstructor.state_dict(), f"cVAE_weights.pth")
            #self.prev_reconstruction_loss = total_loss.item()


        if self.use_tb:
            variance = torch.exp(0.5 * logvar)
            #std_dev = torch.sqrt(variance)

            metrics["cVAE/recon_loss"] = recon_loss.mean().item()
            metrics["cVAE/kl_loss"] = kl_loss.mean().item()
            metrics["cVAE/ssim_loss"] = ssim_loss.mean().item()
            #metrics["cVAE/perceptual_loss"] = perceptual_loss.mean().item()
            if STAGE_1 and LATENT_DYNAMICS:
                metrics["cVAE/latent_dynamic_loss"] = latent_dynamics_loss.mean().item()

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
                    metrics["cVAE/original_images1"] = obs_dynamic[:3, :3, ...]
                    metrics["cVAE/reconstructed_images1"] = obs_reconstructed[:3, :3, ...]

                    metrics["cVAE/original_images2"] = obs_dynamic[:3, 3:6, ...]
                    metrics["cVAE/reconstructed_images2"] = obs_reconstructed[:3, 3:6, ...]

                    metrics["cVAE/original_images3"] = obs_dynamic[:3, 6:9, ...]
                    metrics["cVAE/reconstructed_images3"] = obs_reconstructed[:3, 6:9, ...]
                else:
                    metrics["cVAE/original_images1"] = obs[:3, :3, ...]
                    metrics["cVAE/reconstructed_images1"] = obs_reconstructed[:3, :3, ...]

                    metrics["cVAE/original_images2"] = obs[:3, 3:6, ...]
                    metrics["cVAE/reconstructed_images2"] = obs_reconstructed[:3, 3:6, ...]

                    metrics["cVAE/original_images3"] = obs[:3, 6:9, ...]
                    metrics["cVAE/reconstructed_images3"] = obs_reconstructed[:3, 6:9, ...]

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

        return metrics
    
    def update_decoder(self, obs, next_obs, actions, rewards, step, num_steps):
        """Train the cVAE decoder to reconstruct the input images"""
        metrics = dict()

        obs = obs[:, :self.flat_image_shape].view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        next_obs = next_obs[:, :self.flat_image_shape].view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])

        obs_reconstructed, mu, logvar, _ = self.reconstructor(x=obs, context_actions=actions, context_rewards=rewards)
        kl_weight = min(step / num_steps, 1.0)
        reconstruction_loss, recon_loss, kl_loss = compute_reconstruction_loss(reconstructed=obs_reconstructed, original=obs, mu=mu, logvar=logvar, kl_weight=kl_weight)

        
        # SSIM loss
        ssim_loss = 0.0
        #perceptual_loss = 0.0
        for i in range(0, obs.shape[1] // 3):
            ssim_loss += self.ssim_loss_fn(obs_reconstructed[:, i*3:(i+1)*3, ...], obs[:, i*3:(i+1)*3, ...])
            #with torch.no_grad():
            #    perceptual_loss += self.perception_loss(obs_reconstructed[:, i*3:(i+1)*3, ...], obs[:, i*3:(i+1)*3, ...])

        if WITH_MASKS:
            ssim_loss /= (obs.shape[1] * 2)
        else:
            ssim_loss /= obs.shape[1]
            #perceptual_loss /= obs.shape[1]


        total_loss = reconstruction_loss + ssim_loss



        self.cVAE_opt.zero_grad()
        total_loss.backward()
        self.cVAE_opt.step()


        #if total_loss < self.prev_reconstruction_loss:
            #print(f"Saving cVAE weights at cVAE_weights.pth")
            #torch.save(self.reconstructor.state_dict(), f"cVAE_decoder_weights.pth")
            #self.prev_reconstruction_loss = total_loss.item()


        if self.use_tb:
            variance = torch.exp(0.5 * logvar)

            metrics["cVAE/recon_loss"] = recon_loss.mean().item()
            metrics["cVAE/kl_loss"] = kl_loss.mean().item()
            metrics["cVAE/ssim_loss"] = ssim_loss.mean().item()
            #metrics["cVAE/perceptual_loss"] = perceptual_loss.mean().item()

            metrics["cVAE/reconstruction_loss"] = reconstruction_loss.mean().item()

            metrics["cVAE/distribution_mean"] = mu.mean().item()
            metrics["cVAE/distribution_variance"] = variance.mean().item()
            metrics["cVAE/total_loss"] = total_loss.item()

            if step == 2000 or step % LOG_EVERY == 0:
                metrics["cVAE/original_images1"] = obs[:3, :3, ...]
                metrics["cVAE/decoder_reconstructed_images1"] = obs_reconstructed[:3, :3, ...]

                metrics["cVAE/original_images2"] = obs[:3, 3:6, ...]
                metrics["cVAE/decoder_reconstructed_images2"] = obs_reconstructed[:3, 3:6, ...]

                metrics["cVAE/original_images3"] = obs[:3, 6:9, ...]
                metrics["cVAE/decoder_reconstructed_images3"] = obs_reconstructed[:3, 6:9, ...]

        return metrics

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
            
            next_states = next_obs[:, self.flat_image_shape:]
            next_images = next_obs[:, :self.flat_image_shape]
            next_obs = self.state_encoder(next_states)

        # NOTE: This serves as the sampled augmented entries to update the policies instead of detreminitically augmenting the observations --- Address
        elif STAGE_2:
            # NOTE: Load the cVAE weights and the actor policy weights
            states = obs[:, self.flat_image_shape:]
            images = obs[:, :self.flat_image_shape].view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])
            next_images = next_obs[:, :self.flat_image_shape].view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])

            actions_norm = action / action.norm(dim=-1, keepdim=True)
            reward_norm = reward / reward.norm(dim=-1, keepdim=True)
            sampled_obs = self.reconstructor.sample(context_actions=actions_norm, context_rewards=reward_norm)
            # The actions and rewards are more or less similar for immediate next state
            # TODO: Use latent dynamics to decode the next obs
            # sampled_next_obs = self.reconstructor.sample(context_actions=actions_norm, context_rewards=reward_norm)

            batch_half_size = images.size(0) // 2
            replace_indices = torch.randperm(images.size(0))[:batch_half_size]
            images[replace_indices] = sampled_obs[:batch_half_size]

            # images = torch.cat([images, sampled_obs], dim=0)
            # next_images = torch.cat([next_images, next_images], dim=0)
            # reward_norm = torch.tile(reward_norm, (2, 1))
            # actions_norm = torch.tile(actions_norm, (2, 1))
            
            metrics["cVAE/sampled_images1"] = sampled_obs[:10, :3, ...]
            metrics["cVAE/sampled_images2"] = sampled_obs[:10, 3:6, ...]
            metrics["cVAE/sampled_images3"] = sampled_obs[:10, 6:9, ...]

            # metrics["cVAE/sampled_next_images1"] = sampled_next_obs[:10, :3, ...]
            # metrics["cVAE/sampled_next_images2"] = sampled_next_obs[:10, 3:6, ...]
            # metrics["cVAE/sampled_next_images3"] = sampled_next_obs[:10, 6:9, ...]

            #obs = self.reconstructor.encode(images)
            #next_obs = self.reconstructor.encode(next_images)
            obs = self.image_encoder(images, flatten=True)
            next_obs = self.image_encoder(next_images, flatten=True) # TODO: try the cVAE imag encoder

            #action = torch.tile(action, (2, 1))
            #reward = torch.tile(reward, (2, 1))
            #discount = torch.tile(discount, (2, 1))

            if SHUFFLE:
                indices = torch.randperm(images.size(0))
                images = images[indices].squeeze(1)
                next_images = next_images[indices].squeeze(1)
                action = action[indices]
                reward = reward[indices]
                discount = discount[indices]

        else:
            raise("Just STAGE_1 and STAGE_2")

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        

        # update critic
        metrics_critic = self.update_critic(obs, action, reward, discount, next_obs, step)
        metrics.update(metrics_critic)

        # update actor
        if STAGE_1:
            metrics.update(self.update_actor(obs.detach(), step))
        elif STAGE_2:
            states_feats = None
            if REGULARIZE_ACTOR:
                with torch.no_grad():
                    states_feats = self.state_encoder(states)
                metrics.update(self.update_actor(obs.detach(), step, states_feats=states_feats.detach()))
            else:
                metrics.update(self.update_actor(obs.detach(), step))
        else:
            raise("Just STAGE_1 and STAGE_2")

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        



        # NOTE (important): This needs to be after the updates to the above policies to train properly!!!
        # update reconstruction
        if STAGE_1 and WITH_RECONSTRUCTION:
            actions_norm = action / action.norm(dim=-1, keepdim=True)
            reward_norm = reward / reward.norm(dim=-1, keepdim=True)

            if HIGH_REWARD:
                # try to select the top reward images
                best_indices = torch.argsort(reward, descending=True, dim=0)[:100]
                high_reward_images = images[best_indices].squeeze(1)
                high_reward_images_next = next_images[best_indices].squeeze(1)

                #sampled_obs = torch.cat([next_images, high_reward_images], dim=0)
                high_rewards = reward[best_indices].squeeze(1)
                high_actions = action[best_indices].squeeze(1)
                
                actions_norm = high_actions / high_actions.norm(dim=-1, keepdim=True)
                reward_norm = high_rewards / high_rewards.norm(dim=-1, keepdim=True)
                images = high_reward_images
                next_images = high_reward_images_next

            if AUGMENT:
                # NOTE: AUGMENT (shift and/or noise)
                #images_aug = self.aug_noise(self.aug(images[:, :self.flat_image_shape].view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2]).float()))
                images_aug = self.aug(images[:, :self.flat_image_shape].view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2]).float())
                images_aug = images_aug.view(-1, self.flat_image_shape)
                images = torch.cat([images, images_aug], dim=0)
                actions_norm = torch.cat([actions_norm, actions_norm], dim=0)
                reward_norm = torch.cat([reward_norm, reward_norm], dim=0)


            # NOTE (experiment): Discriminator step
            #discriminator_metrics = self.update_discrimination(obs=images, actions=actions_norm, rewards=reward_norm, step=step, num_steps=num_steps)
            #metrics.update(discriminator_metrics)
            
            # NOTE (experiment): Generator step
            reconstruction_metrics = self.update_reconstruction(obs=images, next_obs=next_images, actions=actions_norm, rewards=reward_norm, step=step, num_steps=num_steps)
            metrics.update(reconstruction_metrics)

            # NOTE: Validate sampled output from Normal
            if step == 2000 or step % LOG_EVERY == 0:
                sampled_obs = self.reconstructor.sample(context_actions=actions_norm, context_rewards=reward_norm)
                metrics["cVAE_val/sampled_images1"] = sampled_obs[:10, :3, ...]
                metrics["cVAE_val/sampled_images2"] = sampled_obs[:10, 3:6, ...]
                metrics["cVAE_val/sampled_images3"] = sampled_obs[:10, 6:9, ...]

        elif STAGE_2:
            if TRAIN_STAGE_2:
                # might not be turned on
                reconstruction_metrics = self.update_decoder(obs=images, next_obs=next_images, actions=action, rewards=reward, step=step, num_steps=num_steps)
                metrics.update(reconstruction_metrics)        

        return metrics

    def save_models(self, suffix=""):
        if STAGE_1:
            torch.save(self.actor.state_dict(), f"actor_weights{suffix}.pth")
            torch.save(self.state_encoder.state_dict(), f"state_encoder_weights{suffix}.pth")
            torch.save(self.reconstructor.state_dict(), f"cVAE_weights{suffix}.pth")
        elif STAGE_2:
            torch.save(self.reconstructor.state_dict(), f"cVAE_decoder_weights{suffix}.pth")
