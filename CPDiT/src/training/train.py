"""Training script for latent diffusion transformer"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import yaml

try:
    import mlflow
    from mlflow.pytorch import autolog

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.models import LatentDiffusionTransformer
from src.data import SatelliteDataset, HIMAWARIDataset
from .config import TrainingConfig


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Training loop manager"""

    def __init__(self, config: TrainingConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = LatentDiffusionTransformer(
            image_channels=config.image_channels,
            latent_dim=config.latent_dim,
            num_transformer_layers=config.num_transformer_layers,
            num_heads=config.num_heads,
            feedforward_dim=config.feedforward_dim,
            num_diffusion_steps=config.num_diffusion_steps,
            dropout=config.dropout,
        ).to(device)

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Setup MLflow if available
        self.mlflow_enabled = HAS_MLFLOW
        if HAS_MLFLOW:
            mlflow.set_tracking_uri(config.tracking_uri)
            mlflow.set_experiment(config.experiment_name)

    def setup_data(self) -> tuple:
        """Setup data loaders"""
        logger.info("Setting up datasets...")

        # Create datasets
        train_datasets = []
        if self.config.train_data_paths:
            train_datasets.append(
                SatelliteDataset(
                    self.config.train_data_paths,
                    context_length=self.config.context_length,
                    forecast_length=self.config.forecast_length,
                )
            )

        if not train_datasets:
            raise ValueError("No training data paths specified")

        train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        # Validation loader
        val_loader = None
        if self.config.val_data_paths:
            val_dataset = SatelliteDataset(
                self.config.val_data_paths,
                context_length=self.config.context_length,
                forecast_length=self.config.forecast_length,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc="Training")
        for context, forecast in pbar:
            context = context.to(self.device)
            forecast = forecast.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            try:
                predictions, _ = self.model(context, forecast)

                # Compute loss
                # Flatten to compare predictions with target
                loss = self.loss_fn(predictions, forecast[:, -predictions.shape[1] :])

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
            except RuntimeError as e:
                logger.warning(f"Skipping batch due to error: {e}")
                continue

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader):
        """Validate model"""
        if val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for context, forecast in tqdm(val_loader, desc="Validation"):
                context = context.to(self.device)
                forecast = forecast.to(self.device)

                try:
                    predictions, _ = self.model(context, forecast)
                    loss = self.loss_fn(predictions, forecast[:, -predictions.shape[1] :])
                    total_loss += loss.item()
                except RuntimeError:
                    continue

        return total_loss / len(val_loader)

    def save_checkpoint(self, epoch: int, val_loss: float = None):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
        }

        path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

        return path

    def train(self, num_epochs: int = None):
        """Full training loop"""
        num_epochs = num_epochs or self.config.num_epochs
        train_loader, val_loader = self.setup_data()

        if self.mlflow_enabled:
            with mlflow.start_run(run_name=self.config.run_name):
                mlflow.log_params(self.config.to_dict())
                self._training_loop(train_loader, val_loader, num_epochs)
        else:
            self._training_loop(train_loader, val_loader, num_epochs)

    def _training_loop(self, train_loader, val_loader, num_epochs):
        """Core training loop"""
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_loss:.6f}")

            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
                logger.info(f"Val Loss: {val_loss:.6f}")

            # Log metrics
            if self.mlflow_enabled:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                if val_loss:
                    mlflow.log_metric("val_loss", val_loss, step=epoch)

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1, val_loss)

                if val_loss and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch + 1, val_loss)
                    logger.info(f"New best model! Val Loss: {val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Load config
    config = TrainingConfig.from_yaml(args.config)

    # Create trainer
    trainer = Trainer(config, device=args.device)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
