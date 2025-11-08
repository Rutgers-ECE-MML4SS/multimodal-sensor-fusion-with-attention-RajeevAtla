"""
Dataset Loading and Preprocessing for Multimodal Sensor Fusion

Provides generic dataset loaders with:
- Modality masking for missing data simulation
- Preprocessing utilities
- Support for multiple datasets (PAMAP2, MHAD, Cooking, Synthetic)
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from itertools import accumulate
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class MultimodalDataset(data.Dataset):
    """
    Generic multimodal dataset for sensor fusion.

    Loads pre-processed features for each modality and handles missing data.
    """

    def __init__(
        self,
        data_dir: str,
        modalities: List[str],
        split: str = "train",
        transform=None,
        modality_dropout: float = 0.0,
        max_shard_cache: int = 4,
    ):
        """
        Args:
            data_dir: Path to dataset directory
            modalities: List of modality names to load
            split: One of ['train', 'val', 'test']
            transform: Optional data augmentation transform
            modality_dropout: Probability of dropping each modality (training only)
            max_shard_cache: Number of manifest shards to keep in RAM simultaneously
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.split = split
        self.transform = transform
        self.modality_dropout = modality_dropout if split == "train" else 0.0
        self.max_shard_cache = max(1, max_shard_cache)

        self.use_manifest = False
        self.data: Dict[str, np.ndarray] = {}
        self.labels: Optional[np.ndarray] = None

        manifest_path = self.data_dir / "splits" / f"{self.split}.txt"
        if manifest_path.exists():
            self._init_from_manifest(manifest_path)
        else:
            self.data, self.labels = self._load_numpy_split()

    def _load_numpy_split(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Load preprocessed data from disk.

        Expected file structure:
        data_dir/
            train/
                modality1.npy  # (N, feature_dim) or (N, seq_len, feature_dim)
                modality2.npy
                labels.npy     # (N,)
            val/
                ...
            test/
                ...
        """
        split_dir = self.data_dir / self.split

        # Load each modality
        data = {}
        for modality in self.modalities:
            modality_file = split_dir / f"{modality}.npy"
            if modality_file.exists():
                data[modality] = np.load(modality_file)
            else:
                raise FileNotFoundError(f"Modality file not found: {modality_file}")

        labels_file = split_dir / "labels.npy"
        if labels_file.exists():
            labels = np.load(labels_file)
        else:
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        return data, labels

    def _init_from_manifest(self, manifest_path: Path) -> None:
        """Initialise dataset backed by sharded CSV manifests."""

        entries = []
        project_root = manifest_path.parents[2] if len(manifest_path.parents) >= 3 else Path(".")
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if "," not in line:
                    raise ValueError(
                        f"Malformed manifest entry '{line}' in {manifest_path}"
                    )
                path_str, rows_str = line.split(",", 1)
                shard_path = Path(path_str)
                if not shard_path.is_absolute():
                    shard_path = (project_root / shard_path).resolve()
                rows = int(rows_str)
                if rows <= 0:
                    continue
                if not shard_path.exists():
                    raise FileNotFoundError(
                        f"Shard referenced in manifest not found: {shard_path}"
                    )
                entries.append({"path": shard_path, "rows": rows})

        if not entries:
            raise ValueError(f"No shards found in manifest {manifest_path}")

        self.use_manifest = True
        self._shard_paths: List[Path] = [e["path"] for e in entries]
        self._shard_rows: List[int] = [e["rows"] for e in entries]
        self._cumulative_rows: List[int] = list(accumulate(self._shard_rows))
        self._total_rows: int = self._cumulative_rows[-1]
        self._shard_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

        # Capture column metadata from the first shard
        sample_columns = pd.read_csv(self._shard_paths[0], nrows=0).columns.tolist()
        self._modality_columns = self._resolve_modality_columns(sample_columns)

    def _resolve_modality_columns(self, columns: List[str]) -> Dict[str, List[str]]:
        """Map requested modalities to the appropriate CSV column subsets."""

        column_set = set(columns)
        mapping: Dict[str, List[str]] = {}
        for modality in self.modalities:
            normalized = modality.lower()
            candidate: List[str] = []
            if normalized in {"heart_rate", "heart", "hr"}:
                if "heart_rate_bpm" in column_set:
                    candidate = ["heart_rate_bpm"]
            else:
                prefix = normalized
                if prefix.startswith("imu_"):
                    prefix = prefix.split("imu_", 1)[1]
                if prefix.endswith("_imu"):
                    prefix = prefix.rsplit("_imu", 1)[0]
                prefix = prefix.replace(" ", "")
                candidate = [col for col in columns if col.startswith(f"{prefix}_")]

            if not candidate:
                raise ValueError(
                    f"Could not resolve modality '{modality}'. "
                    f"Available columns: {columns}"
                )
            mapping[modality] = candidate
        return mapping

    def _locate_shard_index(self, idx: int) -> Tuple[int, int]:
        """Return (shard_idx, row_offset) for the provided global index."""

        shard_idx = bisect.bisect_left(self._cumulative_rows, idx + 1)
        prev_total = self._cumulative_rows[shard_idx - 1] if shard_idx > 0 else 0
        row_offset = idx - prev_total
        return shard_idx, row_offset

    def _get_shard_dataframe(self, shard_idx: int) -> pd.DataFrame:
        """Load (or fetch cached) shard DataFrame."""

        path = self._shard_paths[shard_idx]
        key = str(path)
        if key in self._shard_cache:
            df = self._shard_cache.pop(key)
            self._shard_cache[key] = df
            return df

        df = pd.read_csv(path)
        self._shard_cache[key] = df
        if len(self._shard_cache) > self.max_shard_cache:
            self._shard_cache.popitem(last=False)
        return df

    def _get_manifest_sample(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Retrieve features/label for the given manifest-backed sample."""

        shard_idx, row_offset = self._locate_shard_index(idx)
        df = self._get_shard_dataframe(shard_idx)
        row = df.iloc[row_offset]

        features: Dict[str, torch.Tensor] = {}
        for modality, cols in self._modality_columns.items():
            values = row[cols].to_numpy(dtype=np.float32, copy=True)
            features[modality] = torch.from_numpy(values)

        label = torch.tensor(int(row["activity_id"])).long()
        return features, label

    def __len__(self) -> int:
        if self.use_manifest:
            return self._total_rows
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            features: Dict of {modality_name: tensor}
            label: Class label
            mask: Binary mask indicating available modalities
        """
        if self.use_manifest:
            features, label = self._get_manifest_sample(idx)
        else:
            features = {}
            for modality in self.modalities:
                feat = self.data[modality][idx]
                features[modality] = torch.from_numpy(feat).float()
            label = torch.tensor(self.labels[idx]).long()

        # Apply data augmentation if provided
        if self.transform is not None:
            features = self.transform(features)

        # Create modality availability mask
        mask = torch.ones(len(self.modalities))

        # Apply modality dropout during training
        if self.modality_dropout > 0:
            dropout_mask = torch.rand(len(self.modalities)) > self.modality_dropout
            mask = mask * dropout_mask

            # Ensure at least one modality is available
            if mask.sum() == 0:
                mask[torch.randint(0, len(self.modalities), (1,))] = 1

        return features, label, mask


class SyntheticMultimodalDataset(data.Dataset):
    """
    Synthetic multimodal dataset for quick testing.

    Generates random data with controllable properties.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        num_classes: int = 5,
        modality_dims: Optional[Dict[str, int]] = None,
        sequence_length: int = 100,
        split: str = "train",
        seed: int = 42,
    ):
        """
        Args:
            num_samples: Number of samples to generate
            num_classes: Number of classes
            modality_dims: Dict of {modality_name: feature_dim}
            sequence_length: Length of temporal sequences
            split: Dataset split (affects random seed)
            seed: Random seed for reproducibility
        """
        if modality_dims is None:
            modality_dims = {"sensor1": 32, "sensor2": 32, "sensor3": 32}

        modality_dims = dict(modality_dims)
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.sequence_length = sequence_length

        # Set seed based on split
        split_seeds = {"train": seed, "val": seed + 1, "test": seed + 2}
        np.random.seed(split_seeds.get(split, seed))

        # Generate synthetic data
        self.data = self._generate_data()
        self.labels = np.random.randint(0, num_classes, num_samples)

    def _generate_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic features for each modality."""
        data = {}
        for modality, dim in self.modality_dims.items():
            # Generate sequences with some class-dependent patterns
            data[modality] = np.random.randn(
                self.num_samples, self.sequence_length, dim
            ).astype(np.float32)
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        features = {}
        for modality in self.modalities:
            features[modality] = torch.from_numpy(self.data[modality][idx])

        label = torch.tensor(self.labels[idx]).long()
        mask = torch.ones(len(self.modalities))

        return features, label, mask


def collate_multimodal(batch: List) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for multimodal data.

    Handles variable-length sequences and modality availability.
    """
    features_list, labels_list, masks_list = zip(*batch)

    # Stack features for each modality
    batch_features = {}
    modality_names = features_list[0].keys()

    for modality in modality_names:
        modality_features = [f[modality] for f in features_list]
        batch_features[modality] = torch.stack(modality_features)

    # Stack labels and masks
    batch_labels = torch.stack(labels_list)
    batch_masks = torch.stack(masks_list)

    return batch_features, batch_labels, batch_masks


def create_dataloaders(
    dataset_name: str,
    data_dir: str,
    modalities: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    modality_dropout: float = 0.0,
    **kwargs,
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        dataset_name: Name of dataset ('pamap2', 'mhad', 'cooking', 'synthetic')
        data_dir: Path to dataset directory
        modalities: List of modality names
        batch_size: Batch size
        num_workers: Number of data loading workers
        modality_dropout: Dropout probability for modalities during training
        **kwargs: Additional dataset-specific arguments

    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset_name == "synthetic":
        # Create synthetic datasets
        train_dataset = SyntheticMultimodalDataset(
            num_samples=kwargs.get("num_samples", 10000),
            num_classes=kwargs.get("num_classes", 5),
            modality_dims={m: kwargs.get("modality_dim", 32) for m in modalities},
            split="train",
        )
        val_dataset = SyntheticMultimodalDataset(
            num_samples=kwargs.get("num_samples", 2000) // 5,
            num_classes=kwargs.get("num_classes", 5),
            modality_dims={m: kwargs.get("modality_dim", 32) for m in modalities},
            split="val",
        )
        test_dataset = SyntheticMultimodalDataset(
            num_samples=kwargs.get("num_samples", 2000) // 5,
            num_classes=kwargs.get("num_classes", 5),
            modality_dims={m: kwargs.get("modality_dim", 32) for m in modalities},
            split="test",
        )
    else:
        # Load real datasets
        train_dataset = MultimodalDataset(
            data_dir, modalities, "train", modality_dropout=modality_dropout
        )
        val_dataset = MultimodalDataset(data_dir, modalities, "val")
        test_dataset = MultimodalDataset(data_dir, modalities, "test")

    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_multimodal,
        pin_memory=True,
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multimodal,
        pin_memory=True,
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multimodal,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def simulate_missing_modalities(
    features: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    missing_pattern: Optional[List[int]] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Simulate missing modalities for robustness testing.

    Args:
        features: Dict of modality features
        mask: Current availability mask
        missing_pattern: List of modality indices to keep (None = use mask)

    Returns:
        features: Dict with masked modalities zeroed out
        mask: Updated availability mask
    """
    if missing_pattern is not None:
        # Create new mask based on pattern
        new_mask = torch.zeros_like(mask)
        for idx in missing_pattern:
            new_mask[idx] = 1
        mask = new_mask

    # Zero out features for missing modalities
    modality_names = list(features.keys())
    for i, modality in enumerate(modality_names):
        if mask[i] == 0:
            features[modality] = torch.zeros_like(features[modality])

    return features, mask


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset creation...")

    # Test synthetic dataset
    print("\nCreating synthetic dataset...")
    dataset = SyntheticMultimodalDataset(
        num_samples=100,
        num_classes=5,
        modality_dims={"sensor1": 32, "sensor2": 32, "sensor3": 32},
    )

    print(f"Dataset size: {len(dataset)}")
    features, label, mask = dataset[0]
    print(f"Sample features: {list(features.keys())}")
    print(f"Feature shapes: {[f.shape for f in features.values()]}")
    print(f"Label: {label}")
    print(f"Mask: {mask}")

    # Test dataloader
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name="synthetic",
        data_dir="",
        modalities=["sensor1", "sensor2", "sensor3"],
        batch_size=4,
        num_workers=0,
        num_samples=100,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test batch
    batch_features, batch_labels, batch_masks = next(iter(train_loader))
    print(f"\nBatch features shapes: {[f.shape for f in batch_features.values()]}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Batch masks shape: {batch_masks.shape}")

    print("\n✓ Dataset creation working!")
