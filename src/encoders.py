"""
Modality-Specific Encoders for Sensor Fusion

Implements lightweight encoders suitable for CPU training:
1. SequenceEncoder: For time-series data (IMU, audio, motion capture)
2. FrameEncoder: For frame-based data (video features)
3. SimpleMLPEncoder: For pre-extracted features
"""

from typing import Any, Dict, Optional, cast

import torch
import torch.nn as nn


class SequenceEncoder(nn.Module):
    """
    Encoder for sequential/time-series sensor data.

    Options: 1D CNN, LSTM, GRU, or Transformer
    Output: Fixed-size embedding per sequence
    """

    encoder_type: str
    hidden_dim: int
    output_dim: int
    dropout_layer: nn.Dropout
    rnn: Optional[nn.Module]
    conv_net: Optional[nn.Module]
    pool: Optional[nn.Module]
    input_projection: Optional[nn.Linear]
    transformer: Optional[nn.TransformerEncoder]
    projection: nn.Module

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        encoder_type: str = "lstm",
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for RNN/Transformer
            output_dim: Output embedding dimension
            num_layers: Number of encoder layers
            encoder_type: One of ['lstm', 'gru', 'cnn', 'transformer']
            dropout: Dropout probability
        """
        super().__init__()
        cast_self = cast(Any, self)
        cast_self.encoder_type = encoder_type
        cast_self.hidden_dim = hidden_dim
        cast_self.output_dim = output_dim
        self.dropout_layer = nn.Dropout(dropout)

        cast_self.rnn = None
        cast_self.conv_net = None
        cast_self.pool = None
        cast_self.input_projection = None
        cast_self.transformer = None
        self.projection = nn.Identity()

        if encoder_type == "lstm":
            cast_self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.projection = nn.Linear(hidden_dim, output_dim)

        elif encoder_type == "gru":
            cast_self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.projection = nn.Linear(hidden_dim, output_dim)

        elif encoder_type == "cnn":
            cast_self.conv_net = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            cast_self.pool = nn.AdaptiveAvgPool1d(1)
            self.projection = nn.Linear(hidden_dim, output_dim)

        elif encoder_type == "transformer":
            cast_self.input_projection = nn.Linear(input_dim, hidden_dim)
            nhead = 4 if hidden_dim % 4 == 0 else 1
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dropout=dropout,
                batch_first=True,
            )
            cast_self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            self.projection = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(
        self, sequence: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode variable-length sequences.

        Args:
            sequence: (batch_size, seq_len, input_dim) - input sequence
            lengths: Optional (batch_size,) - actual sequence lengths (for padding)

        Returns:
            encoding: (batch_size, output_dim) - fixed-size embedding
        """
        if sequence.dim() != 3:
            raise ValueError(
                f"Expected 3D input sequence, got shape {sequence.shape}"
            )

        batch_size, seq_len, _ = sequence.shape

        if self.encoder_type in ["lstm", "gru"]:
            rnn_input = sequence
            enforce_lengths = lengths is not None
            if self.rnn is None:
                raise RuntimeError("RNN module not initialized.")

            if enforce_lengths:
                assert lengths is not None
                lengths_cpu = (
                    lengths.to(device=sequence.device).to(torch.int64).cpu()
                )
                packed = nn.utils.rnn.pack_padded_sequence(
                    rnn_input,
                    lengths_cpu,
                    batch_first=True,
                    enforce_sorted=False,
                )
                outputs, hidden = self.rnn(packed)
                outputs, _ = nn.utils.rnn.pad_packed_sequence(
                    outputs, batch_first=True, total_length=seq_len
                )
            else:
                outputs, hidden = self.rnn(rnn_input)

            if self.encoder_type == "lstm":
                hidden_state = hidden[0]
            else:
                hidden_state = hidden

            final_state = hidden_state[-1]  # (batch, hidden_dim)
            encoding = self.projection(self.dropout_layer(final_state))
            return encoding

        if self.encoder_type == "cnn":
            x = sequence.transpose(1, 2).contiguous()
            # Conv1d expects channel-first contiguous tensors; torch.compile enforces strides
            if self.conv_net is None or self.pool is None:
                raise RuntimeError("CNN modules not initialized.")
            x = self.conv_net(x)
            x = self.pool(x).squeeze(-1)
            encoding = self.projection(self.dropout_layer(x))
            return encoding

        if self.encoder_type == "transformer":
            if self.input_projection is None or self.transformer is None:
                raise RuntimeError("Transformer modules not initialized.")
            x = self.input_projection(sequence)
            if lengths is not None:
                lengths = lengths.to(device=sequence.device, dtype=torch.long)
                range_tensor = (
                    torch.arange(seq_len, device=sequence.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                src_key_padding_mask = range_tensor >= lengths.unsqueeze(1)
            else:
                src_key_padding_mask = None

            transformer_output = self.transformer(
                x, src_key_padding_mask=src_key_padding_mask
            )

            if src_key_padding_mask is not None:
                valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
                pooled = transformer_output * valid_mask
                pooled = pooled.sum(dim=1) / (
                    valid_mask.sum(dim=1).clamp_min(1.0)
                )
            else:
                pooled = transformer_output.mean(dim=1)

            encoding = self.projection(self.dropout_layer(pooled))
            return encoding

        raise ValueError(f"Unsupported encoder type: {self.encoder_type}")


class FrameEncoder(nn.Module):
    """
    Encoder for frame-based data (e.g., video features).

    Aggregates frame-level features into video-level embedding.
    """

    temporal_pooling: str
    frame_processor: nn.Sequential
    attention: Optional[nn.Linear]
    projection: nn.Sequential

    def __init__(
        self,
        frame_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        temporal_pooling: str = "attention",
        dropout: float = 0.1,
    ):
        """
        Args:
            frame_dim: Dimension of per-frame features
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            temporal_pooling: How to pool frames ['average', 'max', 'attention']
            dropout: Dropout probability
        """
        super().__init__()
        cast_self = cast(Any, self)
        cast_self.temporal_pooling = temporal_pooling
        self.frame_processor = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        cast_self.attention = None
        if temporal_pooling == "attention":
            cast_self.attention = nn.Linear(hidden_dim, 1)
        elif temporal_pooling in ["average", "max"]:
            # Simple pooling, no learnable parameters needed
            cast_self.attention = None
        else:
            raise ValueError(f"Unknown pooling: {temporal_pooling}")

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, frames: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode sequence of frames.

        Args:
            frames: (batch_size, num_frames, frame_dim) - frame features
            mask: Optional (batch_size, num_frames) - valid frame mask

        Returns:
            encoding: (batch_size, output_dim) - video-level embedding
        """
        if frames.dim() != 3:
            raise ValueError(
                f"Expected 3D frame tensor, got shape {frames.shape}"
            )

        processed = self.frame_processor(frames)
        device = processed.device

        if mask is not None:
            mask = mask.to(device=device, dtype=processed.dtype)

        if self.temporal_pooling == "attention":
            pooled = self.attention_pool(processed, mask)
        elif self.temporal_pooling == "average":
            if mask is None:
                pooled = processed.mean(dim=1)
            else:
                weights = mask.unsqueeze(-1)
                pooled = (processed * weights).sum(dim=1) / (
                    weights.sum(dim=1).clamp_min(1e-8)
                )
        elif self.temporal_pooling == "max":
            if mask is None:
                pooled, _ = processed.max(dim=1)
            else:
                masked_frames = processed.masked_fill(
                    mask.unsqueeze(-1) == 0, float("-inf")
                )
                pooled, _ = masked_frames.max(dim=1)
                pooled = torch.nan_to_num(pooled, nan=0.0, neginf=0.0)
        else:
            raise ValueError(
                f"Unknown pooling strategy: {self.temporal_pooling}"
            )

        encoding = self.projection(pooled)
        return encoding

    def attention_pool(
        self, frames: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool frames using learned attention weights.

        Args:
            frames: (batch_size, num_frames, frame_dim)
            mask: Optional (batch_size, num_frames) - valid frames

        Returns:
            pooled: (batch_size, frame_dim) - attended frame features
        """
        if self.attention is None:
            raise RuntimeError("Attention layer not initialized.")
        scores = self.attention(frames)  # (batch, num_frames, 1)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))

        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        pooled = (weights * frames).sum(dim=1)
        return pooled


class SimpleMLPEncoder(nn.Module):
    """
    Simple MLP encoder for pre-extracted features.

    Use this when working with pre-computed features
    (e.g., ResNet features for images, MFCC for audio).
    """

    encoder: nn.Sequential

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features through MLP.

        Args:
            features: (batch_size, input_dim) - input features

        Returns:
            encoding: (batch_size, output_dim) - encoded features
        """
        if features.dim() != 2:
            raise ValueError(
                f"Expected 2D feature tensor, got shape {features.shape}"
            )
        return self.encoder(features)


def build_encoder(
    modality: str,
    input_dim: int,
    output_dim: int,
    encoder_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Factory function to build appropriate encoder for each modality.

    Args:
        modality: Modality name ('video', 'audio', 'imu', etc.)
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        encoder_config: Optional config dict with encoder hyperparameters

    Returns:
        Encoder module appropriate for the modality
    """
    config: Dict[str, Any] = dict(encoder_config) if encoder_config else {}
    override_type = config.pop("type", None)
    modality_key = modality.lower()

    if override_type == "frame":
        return FrameEncoder(
            frame_dim=input_dim, output_dim=output_dim, **config
        )
    elif override_type == "sequence":
        return SequenceEncoder(
            input_dim=input_dim, output_dim=output_dim, **config
        )
    elif override_type == "mlp":
        return SimpleMLPEncoder(
            input_dim=input_dim, output_dim=output_dim, **config
        )
    else:
        if modality_key in ["video", "frames"]:
            return FrameEncoder(
                frame_dim=input_dim, output_dim=output_dim, **config
            )
        if modality_key in [
            "imu",
            "audio",
            "mocap",
            "accelerometer",
        ] or modality_key.startswith("imu_"):
            return SequenceEncoder(
                input_dim=input_dim, output_dim=output_dim, **config
            )
        # Default to MLP for unknown modalities
        return SimpleMLPEncoder(
            input_dim=input_dim, output_dim=output_dim, **config
        )


if __name__ == "__main__":
    # Test encoders
    print("Testing encoders...")

    batch_size = 4
    seq_len = 100
    input_dim = 64
    output_dim = 128

    # Test SequenceEncoder
    print("\nTesting SequenceEncoder...")
    for enc_type in ["lstm", "gru", "cnn"]:
        try:
            encoder = SequenceEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                encoder_type=enc_type,
            )

            sequence = torch.randn(batch_size, seq_len, input_dim)
            output = encoder(sequence)

            assert output.shape == (batch_size, output_dim)
            print(f"✓ {enc_type} encoder working! Output shape: {output.shape}")

        except NotImplementedError:
            print(f"✗ {enc_type} encoder not implemented yet")
        except Exception as e:
            print(f"✗ {enc_type} encoder error: {e}")

    # Test FrameEncoder
    print("\nTesting FrameEncoder...")
    try:
        num_frames = 30
        frame_dim = 512

        encoder = FrameEncoder(
            frame_dim=frame_dim,
            output_dim=output_dim,
            temporal_pooling="attention",
        )

        frames = torch.randn(batch_size, num_frames, frame_dim)
        output = encoder(frames)

        assert output.shape == (batch_size, output_dim)
        print(f"✓ FrameEncoder working! Output shape: {output.shape}")

    except NotImplementedError:
        print("✗ FrameEncoder not implemented yet")
    except Exception as e:
        print(f"✗ FrameEncoder error: {e}")

    # Test SimpleMLPEncoder
    print("\nTesting SimpleMLPEncoder...")
    try:
        encoder = SimpleMLPEncoder(input_dim=input_dim, output_dim=output_dim)

        features = torch.randn(batch_size, input_dim)
        output = encoder(features)

        assert output.shape == (batch_size, output_dim)
        print(f"✓ SimpleMLPEncoder working! Output shape: {output.shape}")

    except NotImplementedError:
        print("✗ SimpleMLPEncoder not implemented yet")
    except Exception as e:
        print(f"✗ SimpleMLPEncoder error: {e}")
