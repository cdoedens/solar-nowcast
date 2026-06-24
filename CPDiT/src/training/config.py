"""Training configuration and utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import yaml


@dataclass
class TrainingConfig:
    """Training configuration compatible with the staged YAML schema."""

    # Model
    image_channels: int = 3
    image_size: int = 64
    latent_dim: int = 256
    hidden_dim: int = 256
    num_transformer_layers: int = 4
    num_heads: int = 8
    feedforward_dim: int = 1024
    dropout: float = 0.1
    num_diffusion_steps: int = 1000
    denoiser_hidden_dim: int = 512

    # Training stages
    stage: int = 2
    freeze_vae: bool = True
    batch_size: int = 16
    max_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True

    # Data
    context_length: int = 12
    forecast_length: int = 6
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Checkpointing / logging
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    resume_from: Optional[str] = None
    seed: int = 42

    # Tracking
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "CPDiT"
    run_name: str = "baseline"
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load config from a YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create a config object from either the nested YAML structure or a flat dict."""
        if not isinstance(config_dict, dict):
            raise TypeError(f"Expected mapping config, got {type(config_dict)}")

        model_cfg = config_dict.get("model", {})
        data_cfg = config_dict.get("data", {})
        training_cfg = config_dict.get("training", {})
        optimiser_cfg = config_dict.get("optimiser", {})
        logging_cfg = config_dict.get("logging", {})
        transformer_cfg = model_cfg.get("transformer", {})
        diffusion_cfg = model_cfg.get("diffusion", {})

        stage = int(training_cfg.get("stage", 2))
        batch_cfg = training_cfg.get("batch_size", {})
        epoch_cfg = training_cfg.get("max_epochs", {})

        if isinstance(batch_cfg, dict):
            batch_size = batch_cfg.get(f"stage{stage}", batch_cfg.get("stage2", batch_cfg.get("stage1", 16)))
        else:
            batch_size = int(batch_cfg)

        if isinstance(epoch_cfg, dict):
            max_epochs = epoch_cfg.get(f"stage{stage}", epoch_cfg.get("stage2", epoch_cfg.get("stage1", 100)))
        else:
            max_epochs = int(epoch_cfg)

        stage_cfg = optimiser_cfg.get(f"stage{stage}", optimiser_cfg.get("stage2", optimiser_cfg.get("stage1", {})))

        return cls(
            image_channels=int(model_cfg.get("image_channels", 3)),
            image_size=int(model_cfg.get("image_size", 64)),
            latent_dim=int(model_cfg.get("latent_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_transformer_layers=int(transformer_cfg.get("num_layers", 4)),
            num_heads=int(transformer_cfg.get("num_heads", 8)),
            feedforward_dim=int(transformer_cfg.get("feedforward_dim", 1024)),
            dropout=float(transformer_cfg.get("dropout", 0.1)),
            num_diffusion_steps=int(diffusion_cfg.get("num_steps", 1000)),
            denoiser_hidden_dim=int(diffusion_cfg.get("denoiser_hidden_dim", 512)),
            stage=stage,
            freeze_vae=bool(training_cfg.get("freeze_vae", True)),
            batch_size=int(batch_size),
            max_epochs=int(max_epochs),
            learning_rate=float(stage_cfg.get("lr", 1e-4)),
            weight_decay=float(stage_cfg.get("weight_decay", 1e-4)),
            gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 1.0)),
            mixed_precision=bool(training_cfg.get("mixed_precision", True)),
            context_length=int(data_cfg.get("context_length", 12)),
            forecast_length=int(data_cfg.get("forecast_length", 6)),
            num_workers=int(data_cfg.get("num_workers", 4)),
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
            checkpoint_dir=str(training_cfg.get("checkpoint_dir", "outputs/checkpoints")),
            log_dir=str(training_cfg.get("log_dir", "outputs/logs")),
            save_every_n_epochs=int(training_cfg.get("save_every_n_epochs", 5)),
            keep_last_n_checkpoints=int(training_cfg.get("keep_last_n_checkpoints", 3)),
            resume_from=training_cfg.get("resume_from"),
            seed=int(training_cfg.get("seed", 42)),
            tracking_uri=str(config_dict.get("tracking_uri", "http://localhost:5000")),
            experiment_name=str(logging_cfg.get("experiment_name", "CPDiT")),
            run_name=str(logging_cfg.get("project_name", "baseline")),
            device=str(config_dict.get("device", "cuda")),
        )

    def to_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return a flat dictionary of the config values."""
        return asdict(self)

    def to_nested_dict(self) -> Dict[str, Any]:
        """Return a nested dictionary matching the YAML schema."""
        return {
            "model": {
                "image_channels": self.image_channels,
                "image_size": self.image_size,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "transformer": {
                    "num_layers": self.num_transformer_layers,
                    "num_heads": self.num_heads,
                    "feedforward_dim": self.feedforward_dim,
                    "dropout": self.dropout,
                },
                "diffusion": {
                    "num_steps": self.num_diffusion_steps,
                    "denoiser_hidden_dim": self.denoiser_hidden_dim,
                },
            },
            "data": {
                "context_length": self.context_length,
                "forecast_length": self.forecast_length,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "prefetch_factor": self.prefetch_factor,
            },
            "training": {
                "stage": self.stage,
                "freeze_vae": self.freeze_vae,
                "batch_size": {f"stage{self.stage}": self.batch_size},
                "max_epochs": {f"stage{self.stage}": self.max_epochs},
                "gradient_clip_norm": self.gradient_clip_norm,
                "mixed_precision": self.mixed_precision,
                "checkpoint_dir": self.checkpoint_dir,
                "log_dir": self.log_dir,
                "save_every_n_epochs": self.save_every_n_epochs,
                "keep_last_n_checkpoints": self.keep_last_n_checkpoints,
                "resume_from": self.resume_from,
                "seed": self.seed,
            },
            "optimiser": {
                f"stage{self.stage}": {
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                }
            },
            "logging": {
                "experiment_name": self.experiment_name,
                "project_name": self.run_name,
            },
            "device": self.device,
        }
