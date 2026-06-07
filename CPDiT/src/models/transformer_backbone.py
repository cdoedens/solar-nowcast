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

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_len, latent_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

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

    def forward(self, latent_seq, mask=None):
        """
        Args:
            latent_seq: (batch_size, seq_len, latent_dim) - sequence of latent vectors
            mask: Optional attention mask

        Returns:
            encoded: (batch_size, seq_len, latent_dim) - encoded sequence
        """
        seq_len = latent_seq.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max {self.max_seq_len}")

        # Add positional encoding
        pe = self.positional_encoding[:, :seq_len, :].to(latent_seq.device)
        x = latent_seq + pe
        x = self.dropout(x)

        # Apply transformer
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        return encoded
