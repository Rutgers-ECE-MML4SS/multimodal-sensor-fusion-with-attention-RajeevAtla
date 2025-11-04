"""
Attention Mechanisms for Multimodal Fusion

Implements:
1. CrossModalAttention: Attention between different modalities
2. TemporalAttention: Attention over time steps in sequences
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: Modality A attends to Modality B.

    Example: Video features attend to IMU features to incorporate
    relevant motion information at each timestep.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            query_dim: Dimension of query modality features
            key_dim: Dimension of key/value modality features
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal attention.

        Args:
            query: (batch_size, query_dim) - features from modality A
            key: (batch_size, key_dim) - features from modality B
            value: (batch_size, key_dim) - features from modality B
            mask: Optional (batch_size,) - binary mask for valid keys

        Returns:
            attended_features: (batch_size, hidden_dim) - query attended by key/value
            attention_weights: (batch_size, num_heads, 1, 1) - attention scores
        """
        batch_size = query.size(0)
        squeeze_query = False
        squeeze_key = False

        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_query = True
        if key.dim() == 2:
            key = key.unsqueeze(1)
            squeeze_key = True
        if value.dim() == 2:
            value = value.unsqueeze(1)

        q_len = query.size(1)
        k_len = key.size(1)

        q = self.query_proj(query)  # (batch, q_len, hidden_dim)
        k = self.key_proj(key)  # (batch, k_len, hidden_dim)
        v = self.value_proj(value)  # (batch, k_len, hidden_dim)

        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, k_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)  # (batch, num_heads, q_len, head_dim)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, q_len, self.hidden_dim)
        )
        attended = self.out_proj(attended)

        if squeeze_query:
            attended = attended.squeeze(1)
        if squeeze_key:
            attn_weights = attn_weights[:, :, :, :1]
        return attended, attn_weights


class TemporalAttention(nn.Module):
    """
    Temporal attention: Attend over sequence of time steps.

    Useful for: Variable-length sequences, weighting important timesteps
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(
        self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal attention.

        Args:
            sequence: (batch_size, seq_len, feature_dim) - temporal sequence
            mask: Optional (batch_size, seq_len) - binary mask for valid timesteps

        Returns:
            attended_sequence: (batch_size, seq_len, hidden_dim) - attended features
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = sequence.shape

        q = self.query_proj(sequence)
        k = self.key_proj(sequence)
        v = self.value_proj(sequence)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )
        attended = self.out_proj(attended)

        if mask is not None:
            attended = attended * mask.unsqueeze(-1)

        return attended, attn_weights

    def pool_sequence(
        self, sequence: torch.Tensor, attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool sequence to fixed-size representation using attention weights.

        Args:
            sequence: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)

        Returns:
            pooled: (batch_size, hidden_dim) - fixed-size representation
        """
        if attention_weights.dim() != 4:
            raise ValueError(
                f"Expected attention weights with 4 dims, got {attention_weights.shape}"
            )

        # Average across heads and query positions to obtain a distribution over timesteps
        mean_weights = attention_weights.mean(dim=1)  # (batch, seq_len, seq_len)
        pooling_weights = mean_weights.mean(dim=1)  # (batch, seq_len)
        pooling_weights = pooling_weights / (
            pooling_weights.sum(dim=1, keepdim=True) + 1e-8
        )

        pooled = torch.bmm(pooling_weights.unsqueeze(1), sequence).squeeze(1)
        return pooled


class PairwiseModalityAttention(nn.Module):
    """
    Pairwise attention between all modality combinations.

    For M modalities, computes M*(M-1)/2 pairwise attention operations.
    Example: {video, audio, IMU} -> {video<->audio, video<->IMU, audio<->IMU}
    """

    def __init__(
        self,
        modality_dims: dict,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            modality_dims: Dict mapping modality names to feature dimensions
                          Example: {'video': 512, 'audio': 128, 'imu': 64}
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim

        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
                )
                for modality, dim in modality_dims.items()
            }
        )

        attention_layers = {}
        for query_mod in self.modality_names:
            for key_mod in self.modality_names:
                if query_mod == key_mod:
                    continue
                attention_layers[f"{query_mod}_to_{key_mod}"] = CrossModalAttention(
                    query_dim=hidden_dim,
                    key_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
        self.attention_layers = nn.ModuleDict(attention_layers)

    def forward(
        self, modality_features: dict, modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[dict, dict]:
        """
        Apply pairwise attention between all modalities.

        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor: (batch_size, feature_dim)
            modality_mask: (batch_size, num_modalities) - availability mask

        Returns:
            attended_features: Dict of {modality_name: attended_features}
            attention_maps: Dict of {f"{mod_a}_to_{mod_b}": attention_weights}
        """
        if not self.modality_names:
            raise ValueError("No modalities provided for PairwiseModalityAttention.")

        reference_modality = self.modality_names[0]
        batch_size = modality_features[reference_modality].size(0)
        device = modality_features[reference_modality].device
        dtype = modality_features[reference_modality].dtype

        if modality_mask is None:
            modality_mask = torch.ones(
                batch_size, self.num_modalities, device=device, dtype=dtype
            )
        else:
            modality_mask = modality_mask.to(device=device, dtype=dtype)

        projected = {
            modality: self.projections[modality](modality_features[modality].to(device))
            for modality in self.modality_names
        }

        aggregated = {
            modality: [projected[modality]] for modality in self.modality_names
        }
        attention_maps = {}

        for query_mod in self.modality_names:
            for key_mod in self.modality_names:
                if query_mod == key_mod:
                    continue

                attention_key = f"{query_mod}_to_{key_mod}"
                if attention_key not in self.attention_layers:
                    continue

                key_index = self.modality_names.index(key_mod)
                key_mask = (
                    modality_mask[:, key_index] if modality_mask is not None else None
                )

                attended, weights = self.attention_layers[attention_key](
                    projected[query_mod],
                    projected[key_mod],
                    projected[key_mod],
                    mask=key_mask,
                )
                aggregated[query_mod].append(attended)
                attention_maps[attention_key] = weights

        attended_features = {}
        for idx, modality in enumerate(self.modality_names):
            stacked = torch.stack(aggregated[modality], dim=0).mean(dim=0)
            attended_features[modality] = stacked * modality_mask[:, idx].unsqueeze(-1)

        return attended_features, attention_maps


def visualize_attention(
    attention_weights: torch.Tensor, modality_names: list, save_path: str = None
) -> None:
    """
    Visualize attention weights between modalities.

    Args:
        attention_weights: (num_heads, num_queries, num_keys) or similar
        modality_names: List of modality names for labeling
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(attention_weights, torch.Tensor):
        tensor = attention_weights.detach().float().cpu()
    else:
        tensor = torch.as_tensor(attention_weights, dtype=torch.float32)

    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:
        tensor = tensor.mean(dim=0)
    while tensor.dim() > 2:
        tensor = tensor.mean(dim=0)

    heatmap = tensor.numpy()
    if heatmap.ndim != 2:
        heatmap = np.expand_dims(heatmap, axis=0)

    fig, ax = plt.subplots(figsize=(4 + 0.5 * heatmap.shape[1], 4))
    im = ax.imshow(heatmap, cmap="viridis", aspect="auto")

    num_queries, num_keys = heatmap.shape
    query_labels = modality_names[:num_queries]
    key_labels = modality_names[:num_keys]

    ax.set_xticks(np.arange(num_keys))
    ax.set_yticks(np.arange(num_queries))
    ax.set_xticklabels(key_labels, rotation=45, ha="right")
    ax.set_yticklabels(query_labels)
    ax.set_xlabel("Key Modality")
    ax.set_ylabel("Query Modality")
    ax.set_title("Cross-Modal Attention Weights")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    # Simple test
    print("Testing attention mechanisms...")

    batch_size = 4
    query_dim = 512  # e.g., video features
    key_dim = 64  # e.g., IMU features
    hidden_dim = 256
    num_heads = 4

    # Test CrossModalAttention
    print("\nTesting CrossModalAttention...")
    try:
        attn = CrossModalAttention(query_dim, key_dim, hidden_dim, num_heads)

        query = torch.randn(batch_size, query_dim)
        key = torch.randn(batch_size, key_dim)
        value = torch.randn(batch_size, key_dim)

        attended, weights = attn(query, key, value)

        assert attended.shape == (batch_size, hidden_dim)
        print(f"✓ CrossModalAttention working! Output shape: {attended.shape}")

    except NotImplementedError:
        print("✗ CrossModalAttention not implemented yet")
    except Exception as e:
        print(f"✗ CrossModalAttention error: {e}")

    # Test TemporalAttention
    print("\nTesting TemporalAttention...")
    try:
        seq_len = 10
        feature_dim = 128

        temporal_attn = TemporalAttention(feature_dim, hidden_dim, num_heads)
        sequence = torch.randn(batch_size, seq_len, feature_dim)

        attended_seq, weights = temporal_attn(sequence)

        assert attended_seq.shape == (batch_size, seq_len, hidden_dim)
        print(f"✓ TemporalAttention working! Output shape: {attended_seq.shape}")

    except NotImplementedError:
        print("✗ TemporalAttention not implemented yet")
    except Exception as e:
        print(f"✗ TemporalAttention error: {e}")
