import json
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import train


def _base_config(tmp_path, optimizer="adamw", scheduler="cosine"):
    return OmegaConf.create(
        {
            "seed": 0,
            "dataset": {
                "modalities": ["sensor"],
                "name": "synthetic",
                "data_dir": "",
                "batch_size": 2,
                "num_workers": 0,
            },
            "model": {
                "encoders": {},
                "output_dim": 3,
                "fusion_type": "hybrid",
                "hidden_dim": 8,
                "dropout": 0.0,
            },
            "training": {
                "optimizer": optimizer,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "scheduler": scheduler,
                "max_epochs": 1,
                "early_stopping_patience": 1,
                "gradient_clip_norm": 0.0,
                "augmentation": {"modality_dropout": 0.0},
            },
            "experiment": {
                "save_dir": str(tmp_path / "outputs"),
                "name": "exp",
                "save_top_k": 1,
                "log_every_n_steps": 1,
            },
        }
    )


def _fake_batch(batch_size=2, feature_dim=64):
    features = {"sensor": torch.ones(batch_size, feature_dim)}
    labels = torch.zeros(batch_size, dtype=torch.long)
    mask = torch.ones(batch_size, 1)
    return features, labels, mask


def test_lightning_module_steps(tmp_path):
    config = _base_config(tmp_path)
    module = train.MultimodalFusionModule(config)

    batch = _fake_batch()
    loss = module.training_step(batch, 0)
    assert loss.requires_grad

    val_logs = module.validation_step(batch, 0)
    assert "preds" in val_logs

    test_logs = module.test_step(batch, 0)
    assert "confidences" in test_logs


def test_configure_optimizers_branches(tmp_path):
    config_cosine = _base_config(tmp_path)
    module_cosine = train.MultimodalFusionModule(config_cosine)
    result_cosine = module_cosine.configure_optimizers()
    assert "optimizer" in result_cosine and "lr_scheduler" in result_cosine

    config_step = _base_config(tmp_path, scheduler="step")
    module_step = train.MultimodalFusionModule(config_step)
    result_step = module_step.configure_optimizers()
    assert result_step["lr_scheduler"]["scheduler"].__class__.__name__ == "StepLR"

    config_none = _base_config(tmp_path, scheduler="none")
    module_none = train.MultimodalFusionModule(config_none)
    optimizer = module_none.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

    config_bad_opt = _base_config(tmp_path, optimizer="sgd")
    module_bad = train.MultimodalFusionModule(config_bad_opt)
    with pytest.raises(ValueError):
        module_bad.configure_optimizers()

    config_adam = _base_config(tmp_path, optimizer="adam")
    module_adam = train.MultimodalFusionModule(config_adam)
    optim = module_adam.configure_optimizers()
    assert isinstance(optim["optimizer"], torch.optim.Adam)


def test_train_main_invocation(tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    (tmp_path / "outputs").mkdir(exist_ok=True)

    def fake_seed_everything(_seed):
        return _seed

    class DummyDataloader(list):
        def __len__(self):
            return 1

    def fake_create_dataloaders(**kwargs):
        batch = _fake_batch()
        loader = DummyDataloader([batch])
        return loader, loader, loader

    class DummyCheckpoint:
        def __init__(self, *args, **kwargs):
            self.best_model_path = Path("best.ckpt")
            self.best_model_score = torch.tensor(0.1234)

    class DummyLogger:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def fit(self, model, train_loader, val_loader):
            assert len(train_loader) == 1

        def test(self, model, test_loader, ckpt_path=None):
            assert ckpt_path == Path("best.ckpt")

    monkeypatch.setattr(train.pl, "seed_everything", fake_seed_everything)
    monkeypatch.setattr(train, "create_dataloaders", fake_create_dataloaders)
    monkeypatch.setattr(train, "ModelCheckpoint", DummyCheckpoint)
    monkeypatch.setattr(train, "TensorBoardLogger", DummyLogger)
    monkeypatch.setattr(train.pl, "Trainer", DummyTrainer)

    train.main.__wrapped__(config)

    results_path = Path(config.experiment.save_dir) / config.experiment.name / "results.json"
    assert results_path.exists()
    saved = json.loads(results_path.read_text())
    assert "best_model_path" in saved and "best_val_loss" in saved


def test_forward_handles_tuple_output(tmp_path):
    config = _base_config(tmp_path)
    config.model.fusion_type = "late"
    module = train.MultimodalFusionModule(config)
    features, _, mask = _fake_batch()
    module.fusion_model = lambda feats, mask=None: (torch.zeros(features.shape[0], config.model.output_dim), {})
    logits = module.forward(features, mask)
    assert logits.shape[0] == 2
