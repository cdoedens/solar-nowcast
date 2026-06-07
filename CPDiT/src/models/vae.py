"""Variational Autoencoder for latent space compression"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class VariationalAutoencoder(nn.Module):
    """
    VAE for encoding satellite images to latent space.
    Compresses high-dimensional image data into a lower-dimensional latent representation.
    """

    def __init__(self, image_channels=3, latent_dim=256, hidden_dim=256):
        """
        Args:
            image_channels: Number of input image channels
            latent_dim: Dimension of latent space
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim * 4 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 4 * 8 * 8, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, hidden_dim * 4 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim, image_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Encode image to latent space"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """Decode latent vector to image"""
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.hidden_dim * 4, 8, 8)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        """Forward pass with VAE loss computation"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def vae_loss(self, x, x_recon, mu, logvar, beta=0.001):
        """
        Compute VAE loss (reconstruction + KL divergence)
        """
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
