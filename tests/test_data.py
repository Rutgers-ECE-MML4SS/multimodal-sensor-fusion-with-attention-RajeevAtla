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
    )

    train_features, train_labels, train_masks = next(iter(real_loaders[0]))
    assert train_features["mod1"].shape[0] == 3
    assert train_labels.shape == torch.Size([3])
    assert train_masks.shape == torch.Size([3, 2])


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
