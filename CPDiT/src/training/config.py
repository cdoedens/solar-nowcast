"""Training configuration and utilities"""

import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Model
    image_channels: int = 3
    latent_dim: int = 256
    num_transformer_layers: int = 4
    num_heads: int = 8
    feedforward_dim: int = 1024
    num_diffusion_steps: int = 1000
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.0001

    # Data
    context_length: int = 12
    forecast_length: int = 6
    train_data_paths: list = None
    val_data_paths: list = None
    test_data_paths: list = None
    num_workers: int = 4

    # Checkpointing
    save_interval: int = 10
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"

    # MLflow
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "CPDiT"
    run_name: str = "baseline"

    # Device
    device: str = "cuda"
    mixed_precision: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load config from YAML file"""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """Save config to YAML file"""
        with open(yaml_path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
