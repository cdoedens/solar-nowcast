"""Transformer backbone for temporal modeling"""

import torch
import torch.nn as nn
import math


class TransformerBackbone(nn.Module):
    """
    Transformer-based temporal encoder for satellite image sequences.
    Processes latent representations of image sequences.
    """

    def __init__(
        self,
        latent_dim=256,
        num_layers=4,
        num_heads=8,
        feedforward_dim=1024,
        dropout=0.1,
        max_seq_len=64,
    ):
        """
        Args:
            latent_dim: Dimension of input latent vectors
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            feedforward_dim: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len

        # # Positional encoding
        # self.positional_encoding = self._create_positional_encoding(max_seq_len, latent_dim)

        self.register_buffer(
            'positional_encoding',
            self._create_positional_encoding(max_seq_len, latent_dim)
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN: more stable training
            activation='gelu',        # GELU is standard in modern transformers, not ReLU
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.input_projection = nn.Linear(latent_dim, latent_dim)
        self.output_projection = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)


        # Pass time deltas for continuous embedding for irregular timestamps
        self.time_mlp = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )



    def _create_positional_encoding(self, max_len, d_model):
        """Create positional encoding for transformer"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask: position i cannot attend to j > i."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

    def forward(self, latent_seq, timestamps=None, padding_mask=None):
        """
        Args:
            latent_seq:    (B, T, latent_dim)
            timestamps: (B, T, 1) — actual elapsed minutes, normalised
            padding_mask:  (B, T) bool tensor, True = position is padding (ignored)
        """
        if timestamps is not None:
            pe = self.time_mlp(timestamps)
        else:
            positions = torch.arange(seq_len, device=latent_seq.device).unsqueeze(0)
            pe = self.positional_embedding(positions)

        seq_len = latent_seq.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}. "
                f"Increase max_seq_len at construction time, or truncate the input."
            )



        x = self.dropout(latent_seq + pe)

        x = self.input_projection(latent_seq)
        x = self.dropout(x + pe)

        causal_mask = self._create_causal_mask(seq_len, x.device)

        encoded = self.transformer_encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
            )
        encoded = self.norm(encoded)
        return self.output_projection(encoded)
    
    def extra_repr(self):
        '''
        For experiment tracking, it's useful to be able to print a model summary
        '''
        return (f"latent_dim={self.latent_dim}, "
                f"max_seq_len={self.max_seq_len}")


