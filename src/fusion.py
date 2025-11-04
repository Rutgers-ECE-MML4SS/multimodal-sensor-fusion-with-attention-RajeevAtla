"""
Multimodal Fusion Architectures for Sensor Integration

This module implements three fusion strategies:
1. Early Fusion: Concatenate features before processing
2. Late Fusion: Independent processing, combine predictions
3. Hybrid Fusion: Cross-modal attention + learned weighting
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, cast

from attention import CrossModalAttention


class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate encoder outputs and process jointly.

    Pros: Joint representation learning across modalities
    Cons: Requires temporal alignment, sensitive to missing modalities
    """

    modality_names: list[str]
    modality_dims: Dict[str, int]
    fusion: nn.Sequential
    num_classes: int
    hidden_dim: int

    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1,
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
                          Example: {'video': 512, 'imu': 64}
            hidden_dim: Hidden dimension for fusion network
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        dims = dict(modality_dims)
        modality_names = list(dims.keys())
        cast_self = cast(Any, self)
        cast_self.modality_names = modality_names
        cast_self.modality_dims = dims
        cast_self.num_classes = num_classes
        cast_self.hidden_dim = hidden_dim
        concat_dim = sum(dims.values())

        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with early fusion.

        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor shape: (batch_size, feature_dim)
            modality_mask: Binary mask (batch_size, num_modalities)
                          1 = available, 0 = missing

        Returns:
            logits: (batch_size, num_classes)
        """
        if not self.modality_names:
            raise ValueError("No modalities configured for EarlyFusion.")

        first_modality = self.modality_names[0]
        batch_size = modality_features[first_modality].size(0)
        device = modality_features[first_modality].device

        if modality_mask is None:
            modality_mask = torch.ones(
                batch_size,
                len(self.modality_names),
                device=device,
                dtype=modality_features[first_modality].dtype,
            )
        else:
            modality_mask = modality_mask.to(
                device=device, dtype=modality_features[first_modality].dtype
            )

        fused_inputs = []
        for idx, modality in enumerate(self.modality_names):
            if modality not in modality_features:
                raise KeyError(
                    f"Missing features for modality '{modality}' in EarlyFusion forward pass."
                )

            features = modality_features[modality]
            if features.dim() != 2:
                raise ValueError(
                    f"Expected 2D tensor for modality '{modality}', got shape {features.shape}."
                )

            mask = modality_mask[:, idx].unsqueeze(-1)
            fused_inputs.append(features.to(device) * mask)

        concat_features = torch.cat(fused_inputs, dim=1)
        logits = self.fusion(concat_features)
        return logits


class LateFusion(nn.Module):
    """
    Late fusion: Independent classifiers per modality, combine predictions.

    Pros: Handles asynchronous sensors, modular per-modality training
    Cons: Limited cross-modal interaction, fusion only at decision level
    """

    modality_names: list[str]
    modality_dims: Dict[str, int]
    num_modalities: int
    classifiers: nn.ModuleDict
    weight_logits: torch.nn.Parameter
    dropout: nn.Dropout

    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1,
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for per-modality classifiers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        dims = dict(modality_dims)
        modality_names = list(dims.keys())
        cast_self = cast(Any, self)
        cast_self.modality_names = modality_names
        cast_self.num_modalities = len(modality_names)
        cast_self.modality_dims = dims

        classifiers = {}
        for modality, dim in dims.items():
            classifiers[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        self.classifiers = nn.ModuleDict(classifiers)

        self.weight_logits = nn.Parameter(torch.zeros(self.num_modalities))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with late fusion.

        Args:
            modality_features: Dict of {modality_name: features}
            modality_mask: Binary mask for available modalities

        Returns:
            logits: (batch_size, num_classes) - fused predictions
            per_modality_logits: Dict of individual modality predictions
        """
        if not self.modality_names:
            raise ValueError("No modalities configured for LateFusion.")

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

        per_modality_logits: Dict[str, torch.Tensor] = {}
        logits_stack = []

        for idx, modality in enumerate(self.modality_names):
            if modality not in modality_features:
                raise KeyError(
                    f"Missing features for modality '{modality}' in LateFusion forward pass."
                )

            features = modality_features[modality].to(device)
            mask = modality_mask[:, idx].unsqueeze(-1)
            masked_features = features * mask

            logits = self.classifiers[modality](self.dropout(masked_features))
            per_modality_logits[modality] = logits
            logits_stack.append(logits.unsqueeze(1))

        stacked_logits = torch.cat(
            logits_stack, dim=1
        )  # (batch, num_modalities, num_classes)

        base_weights = torch.softmax(self.weight_logits, dim=0)  # (num_modalities,)
        weights = (
            base_weights.unsqueeze(0).expand(batch_size, -1).to(device) * modality_mask
        )
        weight_sums = weights.sum(dim=1, keepdim=True)

        uniform_weights = torch.full_like(weights, 1.0 / self.num_modalities)
        normalized_weights = torch.where(
            weight_sums > 0, weights / (weight_sums + 1e-8), uniform_weights
        )

        fused_logits = (stacked_logits * normalized_weights.unsqueeze(-1)).sum(dim=1)
        return fused_logits, per_modality_logits


class HybridFusion(nn.Module):
    """
    Hybrid fusion: Cross-modal attention + learned fusion weights.

    Pros: Rich cross-modal interaction, robust to missing modalities
    Cons: More complex, higher computation cost

    This is the main focus of the assignment!
    """

    modality_names: list[str]
    num_modalities: int
    hidden_dim: int
    projections: nn.ModuleDict
    attention_modules: nn.ModuleDict
    gating_layers: nn.ModuleDict
    classifier: nn.Sequential
    dropout: nn.Dropout

    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for fusion
            num_classes: Number of output classes
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        dims = dict(modality_dims)
        modality_names = list(dims.keys())
        cast_self = cast(Any, self)
        cast_self.modality_names = modality_names
        cast_self.num_modalities = len(modality_names)
        cast_self.hidden_dim = hidden_dim

        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
                )
                for modality, dim in dims.items()
            }
        )

        attention_modules = {}
        for query_mod in self.modality_names:
            for key_mod in self.modality_names:
                if query_mod == key_mod:
                    continue
                attention_modules[f"{query_mod}_to_{key_mod}"] = CrossModalAttention(
                    query_dim=hidden_dim,
                    key_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
        self.attention_modules = nn.ModuleDict(attention_modules)

        self.gating_layers = nn.ModuleDict(
            {modality: nn.Linear(hidden_dim, 1) for modality in self.modality_names}
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with hybrid fusion.

        Args:
            modality_features: Dict of {modality_name: features}
            modality_mask: Binary mask for available modalities
            return_attention: If True, return attention weights for visualization

        Returns:
            logits: (batch_size, num_classes)
            attention_info: Optional dict with attention weights and fusion weights
        """
        if not self.modality_names:
            raise ValueError("No modalities configured for HybridFusion.")

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

        projected_features: Dict[str, torch.Tensor] = {}
        for idx, modality in enumerate(self.modality_names):
            if modality not in modality_features:
                raise KeyError(
                    f"Missing features for modality '{modality}' in HybridFusion forward pass."
                )
            feats = modality_features[modality].to(device)
            mask = modality_mask[:, idx].unsqueeze(-1)
            projected_features[modality] = self.projections[modality](
                self.dropout(feats * mask)
            )

        aggregated: Dict[str, torch.Tensor] = {}
        attention_maps: Dict[str, torch.Tensor] = {}
        modality_lists = {
            modality: [projected_features[modality]] for modality in self.modality_names
        }

        for query_mod in self.modality_names:
            for key_mod in self.modality_names:
                if query_mod == key_mod:
                    continue
                attention_key = f"{query_mod}_to_{key_mod}"
                if attention_key not in self.attention_modules:
                    continue

                key_index = self.modality_names.index(key_mod)
                key_mask = (
                    modality_mask[:, key_index] if modality_mask is not None else None
                )
                attended, attn_weights = self.attention_modules[attention_key](
                    projected_features[query_mod],
                    projected_features[key_mod],
                    projected_features[key_mod],
                    mask=key_mask,
                )
                modality_lists[query_mod].append(attended)
                attention_maps[attention_key] = attn_weights

        for idx, modality in enumerate(self.modality_names):
            stacked = torch.stack(modality_lists[modality], dim=0).mean(dim=0)
            aggregated[modality] = stacked * modality_mask[:, idx].unsqueeze(-1)

        fusion_weights = self.compute_adaptive_weights(aggregated, modality_mask)
        modality_tensor = torch.stack(
            [aggregated[mod] for mod in self.modality_names], dim=1
        )
        fused_representation = (modality_tensor * fusion_weights.unsqueeze(-1)).sum(
            dim=1
        )
        logits = self.classifier(fused_representation)

        if return_attention:
            attention_info = {
                "attention_maps": attention_maps,
                "fusion_weights": fusion_weights,
            }
            return logits, attention_info
        return logits

    def compute_adaptive_weights(
        self, modality_features: Dict[str, torch.Tensor], modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive fusion weights based on modality availability.

        Args:
            modality_features: Dict of modality features
            modality_mask: (batch_size, num_modalities) binary mask

        Returns:
            weights: (batch_size, num_modalities) normalized fusion weights
        """
        if modality_mask is None:
            raise ValueError("modality_mask must be provided for adaptive weighting.")

        device = modality_mask.device
        modality_mask = modality_mask.to(device=device)

        scores = []
        for modality in self.modality_names:
            if modality not in modality_features:
                raise KeyError(
                    f"Missing aggregated features for modality '{modality}'."
                )
            feat = modality_features[modality].to(device)
            scores.append(self.gating_layers[modality](feat))

        score_tensor = torch.cat(scores, dim=1)  # (batch, num_modalities)
        mask = modality_mask.to(device=device, dtype=score_tensor.dtype)

        masked_scores = score_tensor.masked_fill(mask <= 0, float("-inf"))
        weights = torch.softmax(masked_scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights * mask

        sum_weights = weights.sum(dim=1, keepdim=True)
        mask_sum = mask.sum(dim=1, keepdim=True)
        fallback = torch.where(
            mask_sum > 0,
            mask / (mask_sum + 1e-8),
            torch.full_like(mask, 1.0 / self.num_modalities),
        )
        weights = torch.where(sum_weights > 0, weights / (sum_weights + 1e-8), fallback)
        return weights


# Helper functions


def build_fusion_model(
    fusion_type: str, modality_dims: Dict[str, int], num_classes: int, **kwargs
) -> nn.Module:
    """
    Factory function to build fusion models.

    Args:
        fusion_type: One of ['early', 'late', 'hybrid']
        modality_dims: Dictionary mapping modality names to dimensions
        num_classes: Number of output classes
        **kwargs: Additional arguments for fusion model

    Returns:
        Fusion model instance
    """
    fusion_classes = {
        "early": EarlyFusion,
        "late": LateFusion,
        "hybrid": HybridFusion,
    }

    if fusion_type not in fusion_classes:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    return fusion_classes[fusion_type](
        modality_dims=modality_dims, num_classes=num_classes, **kwargs
    )


if __name__ == "__main__":
    # Simple test to verify implementation
    print("Testing fusion architectures...")

    # Test configuration
    modality_dims = {"video": 512, "imu": 64}
    num_classes = 11
    batch_size = 4

    # Create dummy features
    features = {
        "video": torch.randn(batch_size, 512),
        "imu": torch.randn(batch_size, 64),
    }
    mask = torch.tensor(
        [[1, 1], [1, 0], [0, 1], [1, 1]]
    )  # Different availability patterns

    # Test each fusion type
    for fusion_type in ["early", "late", "hybrid"]:
        print(f"\nTesting {fusion_type} fusion...")
        try:
            model = build_fusion_model(fusion_type, modality_dims, num_classes)

            if fusion_type == "late":
                logits, per_mod_logits = model(features, mask)
            else:
                logits = model(features, mask)

            assert logits.shape == (batch_size, num_classes), (
                f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
            )
            print(f"✓ {fusion_type} fusion working! Output shape: {logits.shape}")

        except NotImplementedError:
            print(f"✗ {fusion_type} fusion not implemented yet")
        except Exception as e:
            print(f"✗ {fusion_type} fusion error: {e}")
