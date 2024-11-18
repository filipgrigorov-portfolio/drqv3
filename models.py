import torch
import torch.nn as nn
import utils

class ImageEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(), 
            
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), 
            
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), 
            
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU()
        )

        self.apply(utils.weight_init)

    def forward(self, obs, flatten=True):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        if flatten:
            h = h.view(h.shape[0], -1)
        return h
    
class ImageDecoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),
            
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),

            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),
            
            nn.ReLU(),
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
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class cVAE(nn.Module):
    def __init__(self, input_shape=(3, 84, 84), latent_dim=32, context_dim=0, freeze_encoder=False):
        super(cVAE, self).__init__()
        
        # Encoder
        self.input_shape = input_shape
        C, H, W = input_shape
        self.encoder = ImageEncoder(obs_shape=input_shape)
        self.flatten = nn.Flatten()

        if freeze_encoder:
            print(f"Freezing encoder's parameters (STAGE_2)")
            for param in self.encoder.parameters():
                if param.requires_grad():
                    param.requires_grad = False

        with torch.no_grad():
            sample = torch.rand((1, input_shape[0], input_shape[1], input_shape[2]))
            sample_output = self.encoder(sample, flatten=False)
            self.last_enc_layer_shape = sample_output.shape
            flat_sample_output = self.flatten(sample_output)
            encoder_out_dim = flat_sample_output.shape[-1]
            self.fc_mu = nn.Linear(encoder_out_dim + context_dim, latent_dim)
            self.fc_logvar = nn.Linear(encoder_out_dim + context_dim, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim + context_dim, encoder_out_dim)
        self.decoder = ImageDecoder(obs_shape=input_shape) #TODO
    
    def forward(self, x, context=None):
        batch_size = x.size(0)
        
        # Encode
        encoded = self.flatten(self.encoder(x, flatten=False))  # Shape: [B, encoder_out_dim]
        if context is not None:
            encoded = torch.cat([encoded, context], dim=-1)
        
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Decode
        if context is not None:
            z = torch.cat([z, context], dim=-1)
        
        decoded = self.fc_decode(z)
        decoded = decoded.view(self.last_enc_layer_shape)  # Reshape to match decoder
        reconstructed = self.decoder(decoded)
        return reconstructed, mu, logvar


# Reconstruction Loss and KL Divergence
def compute_reconstruction_loss(reconstructed, original, mu, logvar):
    # Reconstruction loss (pixel-wise MSE)
    recon_loss = F.mse_loss(reconstructed, original, reduction='mean')
    
    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / original.size(0)
    
    return recon_loss + kl_loss


def test_states_encoder():
    raise NotImplementedError

def test_cVAE_parts():
    obs_shape = (3, 84, 84)
    sample = torch.rand((1, obs_shape[0], obs_shape[1], obs_shape[2]))
    model = cVAE(input_shape=obs_shape)
    reconstructed, mu, logvar = model(sample)
    print(f"reconstructed: {reconstructed.shape}")
    print(f"mu: {mu.shape}")
    print(f"logvar: {logvar.shape}")


if __name__ == "__main__":
    test_cVAE_parts()
