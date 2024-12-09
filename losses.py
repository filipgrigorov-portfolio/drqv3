import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """Structural Similarity Index Measure (SSIM) Loss."""
    # NOTE: SSIM measures the similarity between two images by comparing their luminance, contrast, and structure.

    def __init__(self, window_size=11, channel=3, reduction='mean'):
        """
        Initializes the SSIM loss module.
        Args:
            window_size (int): Size of the Gaussian kernel window.
            channel (int): Number of input channels.
            reduction (str): How to reduce the loss ('mean', 'sum', or 'none').
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.reduction = reduction
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        # Precompute Gaussian kernel as a buffer
        self.register_buffer('gaussian_kernel', self._create_gaussian_window(window_size, channel))

    def _create_gaussian_window(self, window_size, channel, sigma=1.5):
        """
        Create a 2D Gaussian kernel for convolution.
        Args:
            window_size (int): Size of the kernel.
            channel (int): Number of input channels.
            sigma (float): Standard deviation for the Gaussian kernel.
        Returns:
            torch.Tensor: Gaussian kernel of shape [channel, 1, window_size, window_size].
        """
        gauss = torch.tensor(
            [np.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2)) for x in range(window_size)]
        )
        gauss = gauss / gauss.sum()  # Normalize
        kernel_2d = torch.outer(gauss, gauss).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        kernel_2d = kernel_2d.expand(channel, 1, window_size, window_size)  # [C, 1, H, W]
        return kernel_2d

    def forward(self, img1, img2):
        """
        Compute the SSIM loss between two images.
        Args:
            img1 (torch.Tensor): First input image of shape [B, C, H, W].
            img2 (torch.Tensor): Second input image of shape [B, C, H, W].
        Returns:
            torch.Tensor: Scalar SSIM loss.
        """
        # Ensure kernel is on the same device as inputs
        kernel = self.gaussian_kernel.to(img1.device).float()

        # Means
        mu1 = F.conv2d(img1, kernel, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, kernel, padding=self.window_size // 2, groups=self.channel)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Variances and covariance
        sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        # SSIM map
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )

        # Convert to loss
        ssim_loss = 1 - ssim_map

        # Reduce loss
        if self.reduction == 'mean':
            return ssim_loss.mean()
        elif self.reduction == 'sum':
            return ssim_loss.sum()
        else:
            return ssim_loss  # No reduction

def relative_mse_loss(reconstructed, target, epsilon=1e-8):
    """Compute relative MSE loss."""
    relative_error = (reconstructed - target) / (target + epsilon)
    return torch.mean(relative_error ** 2)

# Reconstruction Loss and KL Divergence
def compute_reconstruction_loss(reconstructed, original, mu, logvar, kl_weight=0.01):
    # Reconstruction loss (pixel-wise MSE)
    recon_loss = F.mse_loss(reconstructed, original, reduction='mean')
    #recon_loss = relative_mse_loss(reconstructed, original)
    
    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / original.size(0)
    
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

def test_ssim():
    reconstructed = torch.rand((8, 3, 64, 64)).float()  # Fake reconstructed batch
    original = torch.rand((8, 3, 64, 64)).float()       # Fake ground truth batch
    ssim_loss = SSIMLoss(window_size=11, channel=3, reduction='mean')
    loss = ssim_loss(reconstructed, original)
    print("SSIM Loss:", loss.item())


if __name__ == "__main__":
    test_ssim()
