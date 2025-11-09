import sys
from pathlib import Path

import numpy as np
import runpy
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import data


def _make_dataset_dir(
    tmp_path: Path, modalities: list[str], num_samples: int = 4
) -> str:
    base = tmp_path / "dataset"
    for split in ["train", "val", "test"]:
        split_dir = base / split
        split_dir.mkdir(parents=True, exist_ok=True)
        labels = np.arange(num_samples, dtype=np.int64)
        np.save(split_dir / "labels.npy", labels)
        for modality in modalities:
            features = np.ones((num_samples, 2), dtype=np.float32) * (
                modalities.index(modality) + 1
            )
            np.save(split_dir / f"{modality}.npy", features)
    return str(base)


def _normalize_manifest_prefix(modality: str) -> str:
    prefix = modality.lower()
    if prefix.startswith("imu_"):
        prefix = prefix.split("imu_", 1)[1]
    if prefix.endswith("_imu"):
        prefix = prefix.rsplit("_imu", 1)[0]
    return prefix.replace(" ", "")


def _make_manifest_dataset(
    tmp_path: Path,
    modalities: list[str],
    rows_per_shard: int = 4,
    shards_per_split: int = 2,
) -> str:
    base = tmp_path / "manifest_dataset"
    splits_dir = base / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    columns: list[str] = []
    for modality in modalities:
        normalized = modality.lower()
        if normalized in {"heart_rate", "heart", "hr"}:
            columns.append("heart_rate_bpm")
        else:
            prefix = _normalize_manifest_prefix(modality)
            columns.extend([f"{prefix}_{axis}" for axis in ("x", "y")])
    columns.append("activity_id")

    for split in ["train", "val", "test"]:
        entries: list[str] = []
        for shard_idx in range(shards_per_split):
            shard_path = tmp_path / f"{split}_shard{shard_idx}.pt"
            tensor = torch.arange(
                rows_per_shard * len(columns), dtype=torch.float32
            ).reshape(rows_per_shard, len(columns))
            tensor[:, -1] = float(shard_idx)
            if split == "train" and shard_idx == 0:
                tensor[0, 0] = float("nan")
            torch.save({"data": tensor, "columns": columns}, shard_path)
            entries.append(f"{shard_path},{rows_per_shard}")
        (splits_dir / f"{split}.txt").write_text("\n".join(entries))
    return str(base)


def _save_shard(path: Path, columns: list[str], data: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": data, "columns": columns}, path)


def _write_manifest_file(base: Path, split: str, lines: list[str]) -> Path:
    path = base / "splits" / f"{split}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def test_create_dataloaders_prefetch_factor(tmp_path, monkeypatch):
    modalities = ["mod1"]
    root = _make_dataset_dir(tmp_path, modalities, num_samples=2)

    calls: list[dict] = []

    class DummyLoader(list):
        def __init__(self, *args, **kwargs):
            super().__init__([0])
            calls.append(kwargs)

    monkeypatch.setattr(data.data, "DataLoader", DummyLoader)

    loaders = data.create_dataloaders(
        dataset_name="pamap2",
        data_dir=root,
        modalities=modalities,
        batch_size=2,
        num_workers=2,
        prefetch_factor=3,
    )
    assert len(loaders) == 3
    assert all(call["prefetch_factor"] == 3 for call in calls)


def test_multimodal_dataset_loading_and_dropout(tmp_path, monkeypatch):
    modalities = ["mod1", "mod2"]
    root = _make_dataset_dir(tmp_path, modalities)

    # Custom transform to ensure path executed
    def transform(feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v + 1.0 for k, v in feats.items()}

    dataset = data.MultimodalDataset(
        data_dir=root,
        modalities=modalities,
        split="train",
        transform=transform,
        modality_dropout=1.0,
    )

    assert len(dataset) == 4
    features, label, mask = dataset[0]

    # Transform applied and dropout fallback keeps at least one modality
    for idx, (modality, tensor) in enumerate(features.items()):
        expected = float(
            idx + 2
        )  # Original value (idx+1) plus transform offset
        assert torch.allclose(tensor, torch.full_like(tensor, expected))
    assert label.item() == 0
    assert mask.sum() == 1  # Dropout forced to keep a single modality

    # Trigger missing file branch
    with pytest.raises(FileNotFoundError):
        data.MultimodalDataset(
            data_dir=root,
            modalities=["missing_modality"],
            split="train",
        )

    (tmp_path / "dataset" / "train" / "labels.npy").unlink()
    with pytest.raises(FileNotFoundError):
        data.MultimodalDataset(
            data_dir=root,
            modalities=modalities,
            split="train",
        )


def test_synthetic_dataset_and_collate():
    modalities = {"sensor1": 4, "sensor2": 3}
    synthetic = data.SyntheticMultimodalDataset(
        num_samples=5,
        num_classes=3,
        modality_dims=modalities,
        sequence_length=2,
    )
    default_modalities = data.SyntheticMultimodalDataset(num_samples=2)

    assert len(synthetic) == 5
    features, label, mask = synthetic[0]
    assert set(features.keys()) == set(modalities.keys())
    assert label.shape == torch.Size([])
    assert torch.all(mask == 1)
    assert set(default_modalities.modality_dims.keys()) == {
        "sensor1",
        "sensor2",
        "sensor3",
    }

    batch = [synthetic[i] for i in range(2)]
    collated_features, collated_labels, collated_masks = (
        data.collate_multimodal(batch)
    )

    assert collated_features["sensor1"].shape[0] == 2
    assert collated_labels.shape == torch.Size([2])
    assert collated_masks.shape == torch.Size([2, 2])


def test_create_dataloaders_synthetic_and_real(tmp_path, monkeypatch):
    modalities = ["mod1", "mod2"]
    synthetic_loaders = data.create_dataloaders(
        dataset_name="synthetic",
        data_dir="",
        modalities=modalities,
        batch_size=2,
        num_workers=0,
        num_samples=10,
        modality_dim=3,
        pin_memory=False,
    )

    for loader in synthetic_loaders:
        assert isinstance(loader, DataLoader)
        batch_features, batch_labels, batch_masks = next(iter(loader))
        assert set(batch_features.keys()) == set(modalities)
        assert batch_labels.ndim == 1
        assert batch_masks.ndim == 2
        break

    root = _make_dataset_dir(tmp_path, modalities, num_samples=6)
    real_loaders = data.create_dataloaders(
        dataset_name="pamap2",
        data_dir=root,
        modalities=modalities,
        batch_size=3,
        num_workers=0,
        pin_memory=False,
    )

    train_features, train_labels, train_masks = next(iter(real_loaders[0]))
    assert train_features["mod1"].shape[0] == 3
    assert train_labels.shape == torch.Size([3])
    assert train_masks.shape == torch.Size([3, 2])


def test_manifest_dataset_chunking_and_cache(tmp_path):
    modalities = ["imu_left", "imu_right"]
    root = _make_manifest_dataset(tmp_path, modalities, rows_per_shard=4)

    dataset = data.MultimodalDataset(
        data_dir=root,
        modalities=modalities,
        split="train",
        chunk_size=2,
        prefetch_shards=False,
        max_shard_cache=1,
    )

    assert dataset.use_manifest
    assert len(dataset) == 4  # 2 shards * (rows_per_shard / chunk_size)

    features, label, mask = dataset[0]
    assert label.shape == torch.Size([1])
    assert torch.all(mask == 1)
    assert all(feat.shape[0] == 1 for feat in features.values())
    assert torch.all(torch.isfinite(features["imu_left"]))

    dataset[3]
    assert len(dataset._shard_cache) == 1  # cache evicted oldest shard


def test_create_dataloaders_manifest_path(tmp_path):
    modalities = ["imu_only"]
    root = _make_manifest_dataset(tmp_path, modalities, rows_per_shard=2)

    loaders = data.create_dataloaders(
        dataset_name="pamap2",
        data_dir=root,
        modalities=modalities,
        batch_size=4,
        num_workers=0,
        chunk_size=1,
    )

    train_loader, val_loader, test_loader = loaders
    assert train_loader.batch_size == 1
    assert val_loader.batch_size == 1
    assert test_loader.batch_size == 1

    features, labels, mask = next(iter(train_loader))
    assert labels.shape == torch.Size([1])
    assert mask.shape == torch.Size([1, len(modalities)])
    assert set(features.keys()) == set(modalities)


def test_manifest_relative_paths_and_zero_row_entries(tmp_path):
    data_dir = tmp_path / "rel_manifest"
    shard_dir = tmp_path / "rel_shards"
    columns = ["sensor_x", "sensor_y", "activity_id"]
    shard_path = shard_dir / "train.pt"
    tensor = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
    _save_shard(shard_path, columns, tensor)

    _write_manifest_file(
        data_dir,
        "train",
        [
            "",
            "rel_shards/unused.pt,0",
            "rel_shards/train.pt,2",
        ],
    )

    dataset = data.MultimodalDataset(
        data_dir=str(data_dir),
        modalities=["sensor"],
        split="train",
    )
    assert len(dataset) == 1  # chunk_size defaults to None
    features, label, _ = dataset[0]
    assert label.item() == 0
    assert features["sensor"].shape[1] == 2


def test_manifest_rejects_malformed_entry(tmp_path):
    data_dir = tmp_path / "malformed_manifest"
    _write_manifest_file(data_dir, "train", ["missing_comma_entry"])
    with pytest.raises(ValueError, match="Malformed manifest entry"):
        data.MultimodalDataset(
            data_dir=str(data_dir),
            modalities=["sensor"],
            split="train",
        )


def test_manifest_missing_shard_file_raises(tmp_path):
    data_dir = tmp_path / "missing_shard_manifest"
    _write_manifest_file(data_dir, "train", ["ghost.pt,2"])
    with pytest.raises(FileNotFoundError):
        data.MultimodalDataset(
            data_dir=str(data_dir),
            modalities=["sensor"],
            split="train",
        )


def test_manifest_without_valid_entries_raises(tmp_path):
    data_dir = tmp_path / "empty_manifest"
    _write_manifest_file(data_dir, "train", ["ignored.pt,0"])
    with pytest.raises(ValueError, match="No shards found"):
        data.MultimodalDataset(
            data_dir=str(data_dir),
            modalities=["sensor"],
            split="train",
        )


def test_manifest_requires_activity_column(tmp_path):
    data_dir = tmp_path / "needs_activity"
    shard_path = tmp_path / "activityless.pt"
    columns = ["sensor_x"]
    tensor = torch.ones(1, len(columns))
    _save_shard(shard_path, columns, tensor)
    _write_manifest_file(data_dir, "train", [f"{shard_path},1"])

    with pytest.raises(ValueError, match="activity_id"):
        data.MultimodalDataset(
            data_dir=str(data_dir),
            modalities=["sensor"],
            split="train",
        )


def test_manifest_supports_heart_rate_and_suffix_variants(tmp_path):
    data_dir = tmp_path / "heart_manifest"
    shard_path = tmp_path / "heart.pt"
    columns = ["sensor_x", "sensor_y", "heart_rate_bpm", "activity_id"]
    tensor = torch.tensor(
        [
            [1.0, 2.0, 70.0, 0.0],
            [3.0, 4.0, 72.0, 0.0],
        ]
    )
    _save_shard(shard_path, columns, tensor)
    _write_manifest_file(data_dir, "train", [f"{shard_path},2"])

    dataset = data.MultimodalDataset(
        data_dir=str(data_dir),
        modalities=["imu_sensor", "sensor_imu", "heart_rate"],
        split="train",
        chunk_size=2,
    )

    features, _, _ = dataset[0]
    assert set(features.keys()) == {"imu_sensor", "sensor_imu", "heart_rate"}


def test_manifest_missing_modality_mapping_raises(tmp_path):
    data_dir = tmp_path / "missing_modality_manifest"
    shard_path = tmp_path / "missing_modality.pt"
    columns = ["sensor_x", "sensor_y", "activity_id"]
    tensor = torch.zeros(2, len(columns))
    _save_shard(shard_path, columns, tensor)
    _write_manifest_file(data_dir, "train", [f"{shard_path},2"])

    with pytest.raises(ValueError, match="Could not resolve modality"):
        data.MultimodalDataset(
            data_dir=str(data_dir),
            modalities=["unknown_modality"],
            split="train",
        )


def test_manifest_chunk_label_inconsistency_raises(tmp_path):
    data_dir = tmp_path / "label_inconsistency"
    shard_path = tmp_path / "mixed.pt"
    columns = ["sensor_x", "sensor_y", "activity_id"]
    tensor = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 1.0],
            [5.0, 6.0, 1.0],
            [7.0, 8.0, 1.0],
        ]
    )
    _save_shard(shard_path, columns, tensor)
    _write_manifest_file(data_dir, "train", [f"{shard_path},4"])

    dataset = data.MultimodalDataset(
        data_dir=str(data_dir),
        modalities=["sensor"],
        split="train",
        chunk_size=2,
    )

    with pytest.raises(ValueError, match="Activity id varies"):
        dataset[0]


def test_manifest_dropout_mask_never_all_zero(tmp_path):
    modalities = ["imu_left", "imu_right"]
    root = _make_manifest_dataset(tmp_path, modalities, rows_per_shard=2)

    dataset = data.MultimodalDataset(
        data_dir=root,
        modalities=modalities,
        split="train",
        chunk_size=2,
        modality_dropout=1.0,
    )

    _, _, mask = dataset[0]
    assert int(mask.sum()) == 1


def test_collate_identity_requires_single_sample():
    with pytest.raises(ValueError, match="batch_size=1"):
        data.collate_identity([(None, None, None), (None, None, None)])


def test_numpy_dataset_require_labels_guard(tmp_path):
    modalities = ["mod1"]
    root = _make_dataset_dir(tmp_path, modalities, num_samples=2)
    dataset = data.MultimodalDataset(
        data_dir=root,
        modalities=modalities,
        split="train",
    )
    dataset.labels = None  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="Labels are not loaded"):
        len(dataset)


def test_simulate_missing_modalities():
    features = {
        "mod1": torch.ones(2, 3),
        "mod2": torch.ones(2, 3) * 2,
    }
    mask = torch.tensor([1, 1], dtype=torch.float32)

    updated_features, updated_mask = data.simulate_missing_modalities(
        features.copy(), mask.clone(), missing_pattern=[0]
    )
    assert torch.all(updated_mask == torch.tensor([1.0, 0.0]))
    assert torch.all(updated_features["mod2"] == 0)


def test_data_demo_script_runs(capsys):
    runpy.run_module("data", run_name="__main__")
    output = capsys.readouterr().out
    assert "Dataset size" in output

    # Without pattern, respect existing mask
    features = {
        "mod1": torch.ones(2, 3),
        "mod2": torch.ones(2, 3) * 2,
    }
    mask = torch.tensor([1.0, 0.0])
    updated_features, updated_mask = data.simulate_missing_modalities(
        features, mask
    )
    assert torch.all(updated_mask == mask)
    assert torch.all(updated_features["mod2"] == 0)


def test_multimodal_dataset_validation_split_keeps_modalities(tmp_path):
    modalities = ["mod1", "mod2", "mod3"]
    root = _make_dataset_dir(tmp_path, modalities, num_samples=3)

    val_dataset = data.MultimodalDataset(
        data_dir=root,
        modalities=modalities,
        split="val",
        modality_dropout=0.9,
    )

    _, _, mask = val_dataset[0]
    assert torch.all(mask == 1)


def test_synthetic_dataset_split_seeding():
    train_a = data.SyntheticMultimodalDataset(
        num_samples=8,
        num_classes=3,
        modality_dims={"sensor1": 4},
        sequence_length=5,
        seed=123,
        split="train",
    )
    train_b = data.SyntheticMultimodalDataset(
        num_samples=8,
        num_classes=3,
        modality_dims={"sensor1": 4},
        sequence_length=5,
        seed=123,
        split="train",
    )
    val_dataset = data.SyntheticMultimodalDataset(
        num_samples=8,
        num_classes=3,
        modality_dims={"sensor1": 4},
        sequence_length=5,
        seed=123,
        split="val",
    )

    train_features_a, _, _ = train_a[0]
    train_features_b, _, _ = train_b[0]
    val_features, _, _ = val_dataset[0]

    assert torch.equal(train_features_a["sensor1"], train_features_b["sensor1"])
    assert not torch.equal(train_features_a["sensor1"], val_features["sensor1"])
