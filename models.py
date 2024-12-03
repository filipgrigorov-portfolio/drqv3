import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from torchvision.models import resnet18, vgg16
from torchvision.transforms.functional import normalize

LATENT_ADDITONAL_DIMS = 64

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# class ImageEncoder(nn.Module):
#     def __init__(self, obs_shape):
#         super().__init__()

#         assert len(obs_shape) == 3
#         self.repr_dim = 32 * 35 * 35

#         self.convnet = nn.Sequential(
#             nn.Conv2d(obs_shape[0], 32, 3, stride=2),
#             #nn.ReLU(), 
#             Swish(),
            
#             nn.Conv2d(32, 32, 3, stride=1),
#             #nn.ReLU(), 
#             Swish(),
            
#             nn.Conv2d(32, 32, 3, stride=1),
#             #nn.ReLU(), 
#             Swish(),
            
#             nn.Conv2d(32, 32, 3, stride=1),
#             #nn.ReLU()
#             Swish()
#         )

#         self.apply(utils.weight_init)

#     def forward(self, obs, flatten=True):
#         obs = obs / 255.0 - 0.5
#         h = self.convnet(obs)
#         if flatten:
#             h = h.view(h.shape[0], -1)
#         return h
    
class ResidualBlock(nn.Module):
    """
    A single residual block that maintains the same spatial resolution.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Shortcut connection to match input and output dimensions
        self.shortcut = (nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.relu(x + shortcut)
        return out

class ImageEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            #ResidualBlock(32, 32),
            #nn.ReLU(), 
            Swish(),
            
            nn.Conv2d(32, 32, 3, stride=1),
            #ResidualBlock(32, 32),
            #nn.ReLU(),
            Swish(), 
            
            nn.Conv2d(32, 32, 3, stride=1),
            #ResidualBlock(32, 32),
            #nn.ReLU(), 
            Swish(),
            
            nn.Conv2d(32, 32, 3, stride=1),
            #ResidualBlock(32, 32),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, flatten=True):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        if flatten:
            h = h.view(h.shape[0], -1)
        return h


class ResidualBlockTransConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(ResidualBlockTransConv, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        

        self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        out = self.relu(x + shortcut)
        return out

class ImageDecoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            #ResidualBlockTransConv(32, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),
            #nn.ReLU(),
            Swish(),

            #ResidualBlockTransConv(32, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),
            #nn.ReLU(),
            Swish(),

            #ResidualBlockTransConv(32, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            Swish(),

            #ResidualBlockTransConv(32, 32),
            nn.ConvTranspose2d(32, obs_shape[0], kernel_size=3, stride=2, output_padding=1),
        )

        self.apply(utils.weight_init)

    def forward(self, z):
        return self.convnet(z)

    
class StateEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        self.repr_dim = 32 * 35 * 35
        self.mlp = nn.Sequential(
            nn.Linear(in_features=obs_shape, out_features=16),
            nn.ReLU(),

            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=self.repr_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim), 
            nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
        
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        
            nn.Linear(hidden_dim, action_shape[0])
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), 
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), 
            
            nn.Linear(hidden_dim, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), 
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), 
            
            nn.Linear(hidden_dim, 1)
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
    
class ImageDiscriminator(nn.Module):
    def __init__(self, input_channels=3, conditioning_dim=None):
        super(ImageDiscriminator, self).__init__()
        self.conditioning_dim = conditioning_dim

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Fully connected layer for classification
        with torch.no_grad():
            sample = torch.rand((1, input_channels, 84, 84))
            sample_output = self.conv(sample).flatten(1)
            sample_output_shape = sample_output.shape[-1]
            self.fc = nn.Sequential(
                nn.Linear(sample_output_shape, 1),  # Assuming 64x64 input size; adjust as necessary
                nn.Sigmoid()
            )

        # Optional conditioning
        if self.conditioning_dim is not None:
            self.conditioning_layer = nn.Linear(conditioning_dim, 512)

    def forward(self, x, c=None):
        features = self.conv(x).view(x.size(0), -1)  # Flatten features
        if self.conditioning_dim is not None and c is not None:
            c_emb = self.conditioning_layer(c)
            features += c_emb
        output = self.fc(features)
        return output


class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim):
        super(LatentDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim + 64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)
    

class LatentDynamicsModel(nn.Module):
    """z_t+1 = f(z_t, a_t)"""
    def __init__(self, latent_dim, action_dim):
        super(LatentDynamicsModel, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + LATENT_ADDITONAL_DIMS + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim + LATENT_ADDITONAL_DIMS)
        )
    
    def forward(self, z, action):
        return self.fc(torch.cat([z, action], dim=-1))


class cVAE(nn.Module):
    def __init__(self, input_shape=(3, 84, 84), latent_dim=32, context_actions_dim=0, context_rewards_dim=0, freeze_encoder=False):
        super(cVAE, self).__init__()
        
        # Encoder
        self.input_shape = input_shape
        C, H, W = input_shape
        self.encoder = ImageEncoder(obs_shape=input_shape)
        self.flatten = nn.Flatten()

        if freeze_encoder:
            print(f"Freezing encoder's parameters (STAGE_2)")
            for param in self.encoder.parameters():
                if param.requires_grad:
                    param.requires_grad = False

        with torch.no_grad():
            sample = torch.rand((1, input_shape[0], input_shape[1], input_shape[2]))
            sample_output = self.encoder(sample, flatten=False)
            self.last_enc_layer_shape = sample_output.shape
            flat_sample_output = self.flatten(sample_output)
            encoder_out_dim = flat_sample_output.shape[-1]
            
            # Context
            self.actions_encoder = nn.Sequential(
                nn.Linear(context_actions_dim, 32),
            )
            self.rewards_encoder = nn.Sequential(
                nn.Linear(context_rewards_dim, 32),
            )
            out_context_dim = LATENT_ADDITONAL_DIMS #context_actions_dim + context_rewards_dim #64
            
            #mu and logvar
            self.fc_mu = nn.Linear(encoder_out_dim + out_context_dim, latent_dim)
            self.fc_logvar = nn.Linear(encoder_out_dim + out_context_dim, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim + out_context_dim, encoder_out_dim)
        self.decoder = ImageDecoder(obs_shape=input_shape)

    def encode(self, x, flatten=True):
        encoded = self.encoder(x, flatten=flatten)  # Shape: [B, encoder_out_dim]
        return encoded

    def sample(self, context_actions, context_rewards):
        # Decode
        B = context_actions.size(0)
        z = torch.randn(B, self.fc_mu.out_features).to(context_actions.device)

        z_actions = self.actions_encoder(context_actions)
        z_rewards = self.rewards_encoder(context_rewards)
        z = torch.cat([z, z_actions, z_rewards], dim=-1)
        
        decoded = self.fc_decode(z)
        decoded = decoded.view(-1, self.last_enc_layer_shape[1], self.last_enc_layer_shape[2], self.last_enc_layer_shape[3])  # Reshape to match decoder
        reconstructed = self.decoder(decoded)

        return reconstructed
    
    def forward(self, x, context_actions, context_rewards):
        #batch_size = x.size(0)
        
        # Encode
        encoded = self.flatten(self.encoder(x, flatten=False))  # Shape: [B, encoder_out_dim]
        z_actions = self.actions_encoder(context_actions)
        z_rewards = self.rewards_encoder(context_rewards)
        encoded = torch.cat([encoded, z_actions, z_rewards], dim=-1)
        
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        # Reparameterization trick
        var = torch.exp(0.5 * logvar)
        eps = torch.randn_like(var)
        z = mu + eps * var
        
        # Decode
        z = torch.cat([z, z_actions, z_rewards], dim=-1)
        
        decoded = self.fc_decode(z)
        decoded = decoded.view(-1, self.last_enc_layer_shape[1], self.last_enc_layer_shape[2], self.last_enc_layer_shape[3])  # Reshape to match decoder
        reconstructed = self.decoder(decoded)
        return reconstructed, mu, logvar, z
    

# Utils
def compute_frame_difference(inputs, num_frames):
    """
    Compute absolute differences between consecutive frames.
    Args:
        inputs: [B, C*T, H, W] - Stacked frames
        num_frames: int - Number of frames
    Returns:
        frame_diff: [B, T-1, C, H, W] - Frame differences
    """
    B, C_T, H, W = inputs.shape
    T = num_frames
    C = C_T // T

    # Reshape to [B, T, C, H, W]
    frames = inputs.view(B, T, C, H, W)

    # Compute absolute differences between consecutive frames
    frame_diff = torch.abs(frames[:, 1:] - frames[:, :-1])  # [B, T-1, C, H, W]
    return frame_diff

def create_dynamic_mask(frame_diff, threshold=None, mode="static"):
    """
    Create a mask based on frame differences.
    Args:
        frame_diff: [B, T-1, C, H, W] - Frame differences
        threshold: float or None - Optional threshold for masking
    Returns:
        mask: [B, T-1, 1, H, W] - Binary mask highlighting dynamic regions
    """
    # Aggregate differences across the channel dimension
    dynamic_score = frame_diff.mean(dim=2, keepdim=True)  # [B, T-1, 1, H, W]

    # Apply thresholding or normalize to [0, 1]
    if threshold is not None:
        if mode == "dynamic":
            mask = (dynamic_score > threshold).float()  # Binary mask
        elif mode == "static":
            mask = (dynamic_score <= threshold).float()  # Binary mask (of zeros)
        else:
            return dynamic_score / dynamic_score.max()
    else:
        mask = dynamic_score / dynamic_score.max()  # Soft mask in [0, 1]
    return mask

def mask_inputs_with_dynamic_regions(inputs, mask, num_frames, mode):
    """
    Apply a mask to input frames for encoding.
    Args:
        inputs: [B, C*T, H, W] - Input frames
        mask: [B, T-1, 1, H, W] - Dynamic mask
        num_frames: int - Number of frames
    Returns:
        masked_inputs: Masked inputs
    """
    B, C_T, H, W = inputs.shape
    T = num_frames
    C = C_T // T

    # Reshape inputs to [B, T, C, H, W]
    inputs = inputs.view(B, T, C, H, W)

    if mode == "dynamic":
        # Pad the mask to match the input shape
        padded_mask = torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1)  # [B, T, 1, H, W]
    elif mode == "static":
        # mask is zeros, the rest is ones
        padded_mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)  # [B, T, 1, H, W]

    # Multiply mask with the input
    masked_inputs = inputs * padded_mask
    return masked_inputs.view(B, C_T, H, W)


# TODO: Validate
def compute_kl_divergence_on_behaviour_policy(new_policy, pretrained_policy, images_feats, states_feats, std):
    """Compute KL divergence between new and pretrained policies for given states."""

    new_dist = new_policy(images_feats, std)  # Shape: [batch_size, num_actions]
    with torch.no_grad():
        pretrained_dist = pretrained_policy(states_feats, std) # Detach to avoid backprop through pretrained

    new_samples = new_dist.sample()
    pretrained_samples = pretrained_dist.sample()

    log_probs_new = new_dist.log_prob(new_samples)
    log_probs_pretrained = pretrained_dist.log_prob(pretrained_samples)

    kl_div = log_probs_new - log_probs_pretrained
    kl_div_mean = kl_div.mean() 

    return kl_div_mean



class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:9]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        vgg.eval()

    def forward(self, recon, target):
        # Normalize inputs (VGG expects specific mean/std)
        recon_norm = normalize(recon, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        target_norm = normalize(target, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Extract features and compute MSE
        recon_features = self.feature_extractor(recon_norm)
        target_features = self.feature_extractor(target_norm)
        return F.mse_loss(recon_features, target_features)


# def test_image_encoders():
#     obs_shape = (3, 84, 84)
#     sample = torch.rand((1, obs_shape[0], obs_shape[1], obs_shape[2]))

#     model = ImageEncoder(obs_shape=obs_shape)
#     encoded = model(sample, flatten=False)
#     print(f"encoded: {encoded.shape}")

#     model = ResNetEncoder(obs_shape=obs_shape)
#     encoded = model(sample, flatten=False)
#     print(f"encoded: {encoded.shape}")

def test_cVAE_parts():
    obs_shape = (3, 84, 84)
    sample = torch.rand((1, obs_shape[0], obs_shape[1], obs_shape[2]))
    sample_action = torch.rand((1, 1))
    sample_reward = torch.rand((1, 1))
    model = cVAE(input_shape=obs_shape, context_actions_dim=1, context_rewards_dim=1)
    reconstructed, mu, logvar, z = model(sample, sample_action, sample_reward)
    print(f"reconstructed: {reconstructed.shape}")
    print(f"mu: {mu.shape}")
    print(f"logvar: {logvar.shape}")
    print(f"z: {z.shape}")


if __name__ == "__main__":
    test_cVAE_parts()
