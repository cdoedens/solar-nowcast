"""Latent Diffusion Transformer for satellite image forecasting"""

import torch
import torch.nn as nn
from .vae import VariationalAutoencoder
from .transformer_backbone import TransformerBackbone


class LatentDiffusionTransformer(nn.Module):
    """
    Latent Diffusion Transformer that combines:
    1. VAE for image compression to latent space
    2. Transformer for temporal modeling
    3. Diffusion process for temporal prediction
    """

    def __init__(
        self,
        image_channels=3,
        latent_dim=256,
        num_transformer_layers=4,
        num_heads=8,
        feedforward_dim=1024,
        num_diffusion_steps=1000,
        dropout=0.1,
    ):
        """
        Args:
            image_channels: Number of input image channels
            latent_dim: Dimension of latent space
            num_transformer_layers: Number of transformer layers
            num_heads: Number of attention heads
            feedforward_dim: Feedforward dimension in transformer
            num_diffusion_steps: Number of diffusion steps
            dropout: Dropout rate
        """
        super().__init__()

        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.num_diffusion_steps = num_diffusion_steps

        # Components
        self.vae = VariationalAutoencoder(
            image_channels=image_channels, latent_dim=latent_dim
        )
        self.transformer = TransformerBackbone(
            latent_dim=latent_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

        # Diffusion noise scheduling
        self.register_buffer("betas", self._linear_beta_schedule(num_diffusion_steps))
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def _linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        """Linear schedule for noise variance"""
        return torch.linspace(start, end, timesteps)

    def encode_images(self, images):
        """
        Encode batch of images to latent space.
        
        Args:
            images: (batch_size, seq_len, channels, height, width)
            
        Returns:
            latent_codes: (batch_size, seq_len, latent_dim)
        """
        batch_size, seq_len = images.shape[:2]
        # Reshape to (batch_size * seq_len, channels, height, width)
        images_flat = images.view(batch_size * seq_len, *images.shape[2:])

        # Encode
        mu, logvar = self.vae.encode(images_flat)

        # Reshape back to (batch_size, seq_len, latent_dim)
        latent_codes = mu.view(batch_size, seq_len, self.latent_dim)
        return latent_codes

    def decode_latents(self, latent_codes):
        """
        Decode latent codes back to images.
        
        Args:
            latent_codes: (batch_size, seq_len, latent_dim)
            
        Returns:
            images: (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len = latent_codes.shape[:2]
        # Reshape to (batch_size * seq_len, latent_dim)
        latent_flat = latent_codes.view(batch_size * seq_len, self.latent_dim)

        # Decode
        images_flat = self.vae.decode(latent_flat)

        # Reshape back to (batch_size, seq_len, channels, height, width)
        images = images_flat.view(batch_size, seq_len, *images_flat.shape[1:])
        return images

    def add_noise(self, x, timesteps):
        """
        Add noise to latent codes according to diffusion process.
        
        Args:
            x: Clean latent codes
            timesteps: Diffusion timesteps
            
        Returns:
            noisy: Noisy latent codes
            noise: The noise added
        """
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])

        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod.shape) < len(x.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)

        noisy = sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
        return noisy, noise

    def forward(self, context_images, target_images=None):
        """
        Forward pass for training or inference.
        
        Args:
            context_images: (batch_size, context_len, channels, height, width)
            target_images: (batch_size, forecast_len, channels, height, width) - optional
            
        Returns:
            predictions: Forecasted latent codes or images
        """
        # Encode context images
        context_latents = self.encode_images(context_images)

        # Process through transformer
        encoded_context = self.transformer(context_latents)

        # Use last frame as condition for next prediction
        last_latent = encoded_context[:, -1:, :]  # (batch_size, 1, latent_dim)

        # Predict next frames
        predictions = self.pred_head(last_latent)

        return predictions, encoded_context

    def sample(self, context_images, num_forecast_steps, num_samples=1):
        """
        Generate forecast samples using diffusion.
        
        Args:
            context_images: (batch_size, context_len, channels, height, width)
            num_forecast_steps: Number of future timesteps to forecast
            num_samples: Number of samples to generate per context
            
        Returns:
            samples: Generated forecast images
        """
        with torch.no_grad():
            context_latents = self.encode_images(context_images)
            encoded_context = self.transformer(context_latents)

            # Start from noise
            batch_size = context_images.shape[0]
            x = torch.randn(
                batch_size, num_forecast_steps, self.latent_dim, device=context_images.device
            )

            # Reverse diffusion process
            for t in range(self.num_diffusion_steps - 1, -1, -1):
                timesteps = torch.full((batch_size,), t, device=context_images.device)
                predicted_noise = self.pred_head(x)
                
                # Update x based on predicted noise
                alpha = self.betas[t]
                x = (x - alpha * predicted_noise) / torch.sqrt(1.0 - alpha)

            # Decode latents to images
            samples = self.decode_latents(x)
            return samples
