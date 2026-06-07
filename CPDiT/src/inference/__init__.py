"""Inference module for generating satellite image forecasts"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import logging

from src.models import LatentDiffusionTransformer
from src.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class Forecaster:
    """Inference wrapper for satellite image forecasting"""

    def __init__(
        self,
        model: LatentDiffusionTransformer,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model: Initialized LatentDiffusionTransformer model
            checkpoint_path: Path to saved checkpoint
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.device = device

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Checkpoint loaded successfully")

    @torch.no_grad()
    def forecast(
        self,
        context_images: torch.Tensor,
        num_steps: int = 6,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate forecast for satellite images.

        Args:
            context_images: (batch_size, context_len, channels, height, width)
            num_steps: Number of future timesteps to forecast
            num_samples: Number of samples to generate

        Returns:
            forecasts: (batch_size, num_steps, channels, height, width)
        """
        context_images = context_images.to(self.device)
        
        # Generate samples using diffusion
        samples = self.model.sample(context_images, num_steps, num_samples)
        
        return samples.cpu()

    @torch.no_grad()
    def forecast_deterministic(
        self,
        context_images: torch.Tensor,
        num_steps: int = 6,
    ) -> torch.Tensor:
        """
        Generate deterministic forecast (mean prediction).

        Args:
            context_images: (batch_size, context_len, channels, height, width)
            num_steps: Number of future timesteps to forecast

        Returns:
            predictions: (batch_size, num_steps, channels, height, width)
        """
        context_images = context_images.to(self.device)

        # Encode context
        context_latents = self.model.encode_images(context_images)

        # Process through transformer
        encoded_context = self.model.transformer(context_latents)

        # Predict future latents (simplified approach)
        predictions_list = []
        last_latent = encoded_context[:, -1:, :]

        for _ in range(num_steps):
            next_pred = self.model.pred_head(last_latent)
            predictions_list.append(next_pred)
            last_latent = next_pred

        # Stack predictions
        predicted_latents = torch.cat(predictions_list, dim=1)

        # Decode back to images
        forecast_images = self.model.decode_latents(predicted_latents)

        return forecast_images.cpu()

    def forecast_sequence(
        self,
        context_images: torch.Tensor,
        num_steps: int = 6,
        autoregressive: bool = False,
    ) -> torch.Tensor:
        """
        Generate extended forecast sequence.

        Args:
            context_images: Initial context images
            num_steps: Total number of steps to forecast
            autoregressive: If True, use previous predictions as context

        Returns:
            full_sequence: (batch_size, context_len + num_steps, channels, height, width)
        """
        all_frames = [context_images]

        if autoregressive:
            current_context = context_images
            for _ in range(0, num_steps, context_images.shape[1]):
                forecast = self.forecast_deterministic(current_context, context_images.shape[1])
                all_frames.append(forecast)
                current_context = forecast
        else:
            forecast = self.forecast_deterministic(context_images, num_steps)
            all_frames.append(forecast)

        return torch.cat(all_frames, dim=1)


class BatchPredictor:
    """Batch prediction handler for multiple images"""

    def __init__(self, forecaster: Forecaster, batch_size: int = 32):
        self.forecaster = forecaster
        self.batch_size = batch_size

    def predict_batch(
        self,
        data_loader,
        num_forecast_steps: int = 6,
        return_context: bool = True,
    ) -> List[np.ndarray]:
        """
        Predict on entire dataset.

        Args:
            data_loader: DataLoader with batches
            num_forecast_steps: Steps to forecast
            return_context: Include context in output

        Returns:
            List of prediction arrays
        """
        predictions = []

        for context, forecast in data_loader:
            pred = self.forecaster.forecast_deterministic(context, num_forecast_steps)

            if return_context:
                full_sequence = torch.cat([context, pred], dim=1)
                predictions.append(full_sequence.numpy())
            else:
                predictions.append(pred.numpy())

        return predictions


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cuda"
) -> Tuple[LatentDiffusionTransformer, Forecaster]:
    """
    Load model and create forecaster from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on

    Returns:
        model, forecaster: Tuple of model and forecaster
    """
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint["config"]
    config = TrainingConfig(**config_dict)

    # Create model
    model = LatentDiffusionTransformer(
        image_channels=config.image_channels,
        latent_dim=config.latent_dim,
        num_transformer_layers=config.num_transformer_layers,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        num_diffusion_steps=config.num_diffusion_steps,
        dropout=config.dropout,
    )

    # Create forecaster
    forecaster = Forecaster(model, checkpoint_path=checkpoint_path, device=device)

    return model, forecaster
