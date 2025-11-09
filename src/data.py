"""
Dataset Loading and Preprocessing for Multimodal Sensor Fusion

Provides generic dataset loaders with:
- Modality masking for missing data simulation
- Preprocessing utilities
- Support for multiple datasets (PAMAP2, MHAD, Cooking, Synthetic)
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
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
        prefetch_shards: bool = True,
        chunk_size: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Path to dataset directory
            modalities: List of modality names to load
            split: One of ['train', 'val', 'test']
            transform: Optional data augmentation transform
            modality_dropout: Probability of dropping each modality (training only)
            prefetch_shards: Load all manifest shards into memory if True
            max_shard_cache: Number of manifest shards to keep in RAM simultaneously
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.split = split
        self.transform = transform
        self.modality_dropout = modality_dropout if split == "train" else 0.0
        self.prefetch_shards = prefetch_shards
        self.max_shard_cache = max(1, max_shard_cache)
        self.chunk_size = chunk_size

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
                raise FileNotFoundError(
                    f"Modality file not found: {modality_file}"
                )

        labels_file = split_dir / "labels.npy"
        if labels_file.exists():
            labels = np.load(labels_file)
        else:
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        return data, labels

    def _init_from_manifest(self, manifest_path: Path) -> None:
        """Initialise dataset backed by sharded tensor manifests."""

        entries = []
        project_root = (
            manifest_path.parents[2]
            if len(manifest_path.parents) >= 3
            else Path(".")
        )
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

        sample_payload = torch.load(entries[0]["path"])
        columns = list(sample_payload["columns"])
        self._column_to_index = {name: idx for idx, name in enumerate(columns)}
        modality_columns = self._resolve_modality_columns(columns)
        self._modality_column_indices = {
            modality: [self._column_to_index[col] for col in cols]
            for modality, cols in modality_columns.items()
        }
        self._modality_index_tensors = {
            modality: torch.tensor(indices, dtype=torch.long)
            for modality, indices in self._modality_column_indices.items()
        }
        if "activity_id" not in self._column_to_index:
            raise ValueError("activity_id column missing from tensor shards.")
        self._activity_col_index = self._column_to_index["activity_id"]

        self.use_manifest = True
        self._shard_paths: List[Path] = [e["path"] for e in entries]
        self._shard_rows: List[int] = [e["rows"] for e in entries]
        self._total_rows: int = len(self._shard_rows)
        self._shard_cache: OrderedDict[str, dict] = OrderedDict()
        self._chunks: List[Tuple[int, int, int]] = self._build_chunks()

        if self.prefetch_shards:
            for path in self._shard_paths:
                payload = torch.load(path)
                self._shard_cache[str(path)] = payload
            self.max_shard_cache = len(self._shard_paths)
        else:
            self.max_shard_cache = max(1, self.max_shard_cache)

    def _resolve_modality_columns(
        self, columns: List[str]
    ) -> Dict[str, List[str]]:
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
                candidate = [
                    col for col in columns if col.startswith(f"{prefix}_")
                ]

            if not candidate:
                raise ValueError(
                    f"Could not resolve modality '{modality}'. "
                    f"Available columns: {columns}"
                )
            mapping[modality] = candidate
        return mapping

    def _build_chunks(self) -> List[Tuple[int, int, int]]:
        """Return list of (shard_idx, start_row, end_row) slices."""

        chunks: List[Tuple[int, int, int]] = []
        for shard_idx, rows in enumerate(self._shard_rows):
            if self.chunk_size is None:
                chunks.append((shard_idx, 0, rows))
                continue
            start = 0
            while start < rows:
                end = min(start + self.chunk_size, rows)
                chunks.append((shard_idx, start, end))
                start = end
        return chunks

    def _get_shard_data(self, shard_idx: int) -> dict:
        """Load (or fetch cached) shard tensor payload."""

        path = self._shard_paths[shard_idx]
        key = str(path)
        if key in self._shard_cache:
            payload = self._shard_cache.pop(key)
            self._shard_cache[key] = payload
            return payload

        payload = torch.load(path)
        self._shard_cache[key] = payload
        if (
            not self.prefetch_shards
            and len(self._shard_cache) > self.max_shard_cache
        ):
            self._shard_cache.popitem(last=False)
        return payload

    def _require_labels(self) -> np.ndarray:
        """Return loaded labels, raising if they are unavailable."""

        if self.labels is None:
            raise RuntimeError(
                "Labels are not loaded for this dataset split."
            )
        return self.labels

    def __len__(self) -> int:
        if self.use_manifest:
            return len(self._chunks)
        labels = self._require_labels()
        return len(labels)

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
            shard_idx, start, end = self._chunks[idx]
            payload = self._get_shard_data(shard_idx)
            batch = payload["data"][start:end]
            label_values = batch[:, self._activity_col_index]
            label_value = label_values[0].item()
            if not torch.all(label_values == label_values[0]):
                raise ValueError("Activity id varies within shard chunk.")
            features = {}
            for modality, index_tensor in self._modality_index_tensors.items():
                seq = batch.index_select(1, index_tensor).clone().float()
                features[modality] = seq.unsqueeze(0)
            label = torch.tensor([int(label_value)]).long()
        else:
            features = {}
            for modality in self.modalities:
                feat = self.data[modality][idx]
                features[modality] = torch.from_numpy(feat).float()
            labels = self._require_labels()
            label = torch.tensor(labels[idx]).long()

        # Apply data augmentation if provided
        if self.transform is not None:
            features = self.transform(features)

        # Create modality availability mask
        if self.use_manifest:
            mask = torch.ones(label.shape[0], len(self.modalities))
        else:
            mask = torch.ones(len(self.modalities))

        # Apply modality dropout during training
        if self.modality_dropout > 0:
            dropout_mask = (
                torch.rand(len(self.modalities)) > self.modality_dropout
            )
            if self.use_manifest:
                mask = mask * dropout_mask.unsqueeze(0)
            else:
                mask = mask * dropout_mask

            # Ensure at least one modality is available
            if mask.sum() == 0:
                if self.use_manifest:
                    mask[:, torch.randint(0, len(self.modalities), (1,))] = 1
                else:
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


def collate_identity(batch: List):
    """Return the single manifest chunk without stacking."""

    if len(batch) != 1:
        raise ValueError("Manifest dataloader expects batch_size=1.")
    return batch[0]


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
    chunk_size = kwargs.get("chunk_size")

    if dataset_name == "synthetic":
        # Create synthetic datasets
        train_dataset = SyntheticMultimodalDataset(
            num_samples=kwargs.get("num_samples", 10000),
            num_classes=kwargs.get("num_classes", 5),
            modality_dims={
                m: kwargs.get("modality_dim", 32) for m in modalities
            },
            split="train",
        )
        val_dataset = SyntheticMultimodalDataset(
            num_samples=kwargs.get("num_samples", 2000) // 5,
            num_classes=kwargs.get("num_classes", 5),
            modality_dims={
                m: kwargs.get("modality_dim", 32) for m in modalities
            },
            split="val",
        )
        test_dataset = SyntheticMultimodalDataset(
            num_samples=kwargs.get("num_samples", 2000) // 5,
            num_classes=kwargs.get("num_classes", 5),
            modality_dims={
                m: kwargs.get("modality_dim", 32) for m in modalities
            },
            split="test",
        )
    else:
        # Load real datasets
        train_dataset = MultimodalDataset(
            data_dir,
            modalities,
            "train",
            modality_dropout=modality_dropout,
            prefetch_shards=True,
            chunk_size=chunk_size,
        )
        val_dataset = MultimodalDataset(
            data_dir,
            modalities,
            "val",
            prefetch_shards=True,
            chunk_size=chunk_size,
        )
        test_dataset = MultimodalDataset(
            data_dir,
            modalities,
            "test",
            prefetch_shards=True,
            chunk_size=chunk_size,
        )

    def _build_loader(dataset, shuffle: bool) -> data.DataLoader:
        persistent_workers = num_workers > 0
        if getattr(dataset, "use_manifest", False):
            loader_batch_size = 1
            collate_fn = collate_identity
        else:
            loader_batch_size = batch_size
            collate_fn = collate_multimodal
        return data.DataLoader(
            dataset,
            batch_size=loader_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            persistent_workers=persistent_workers,
        )

    train_loader = _build_loader(train_dataset, shuffle=True)
    val_loader = _build_loader(val_dataset, shuffle=False)
    test_loader = _build_loader(test_dataset, shuffle=False)

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
    print(
        f"\nBatch features shapes: {[f.shape for f in batch_features.values()]}"
    )
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Batch masks shape: {batch_masks.shape}")

    print("\n✓ Dataset creation working!")
