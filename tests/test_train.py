import copy
import json
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
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
                "enable_compile": False,
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
    assert (
        result_step["lr_scheduler"]["scheduler"].__class__.__name__ == "StepLR"
    )

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

    results_path = (
        Path(config.experiment.save_dir)
        / config.experiment.name
        / "results.json"
    )
    assert results_path.exists()
    saved = json.loads(results_path.read_text())
    assert "best_model_path" in saved and "best_val_loss" in saved


def test_forward_handles_tuple_output(tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    config.model.fusion_type = "late"

    class TupleFusion(nn.Module):
        def forward(self, feats, mask=None):
            batch_size = next(iter(feats.values())).shape[0]
            logits = torch.zeros(batch_size, config.model.output_dim)
            return logits, {"sensor": logits}

    monkeypatch.setattr(
        train,
        "build_fusion_model",
        lambda *args, **kwargs: TupleFusion(),
    )

    module = train.MultimodalFusionModule(config)
    features, _, mask = _fake_batch()
    logits = module.forward(features, mask)
    assert logits.shape[0] == 2


def test_training_step_logs_metrics(tmp_path):
    config = _base_config(tmp_path)
    module = train.MultimodalFusionModule(config)
    captured = {}

    def fake_log(name, value, **kwargs):
        captured.setdefault(name, []).append((value, kwargs))

    module._log_metric = fake_log  # type: ignore[assignment]

    batch = _fake_batch()
    loss = module.training_step(batch, 0)
    assert loss.requires_grad
    assert "train/loss" in captured
    assert "train/acc" in captured
    assert captured["train/loss"][0][0] is loss
    assert captured["train/acc"][0][1]["on_epoch"]


def test_cosine_scheduler_parameters(tmp_path):
    config = _base_config(tmp_path)
    module = train.MultimodalFusionModule(config)
    optim_config = module.configure_optimizers()
    scheduler = optim_config["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == config.training.max_epochs
    assert scheduler.eta_min == pytest.approx(
        config.training.learning_rate / 100
    )

    config_step = _base_config(tmp_path, scheduler="step")
    module_step = train.MultimodalFusionModule(config_step)
    step_config = module_step.configure_optimizers()
    step_scheduler = step_config["lr_scheduler"]["scheduler"]
    assert isinstance(step_scheduler, torch.optim.lr_scheduler.StepLR)
    assert step_scheduler.step_size == 30
    assert step_scheduler.gamma == 0.1


def test_compile_helpers_cache_and_reload(monkeypatch):
    train._COMPILED_MODULE_CACHE.clear()
    base_module = nn.Linear(4, 2)

    clone = train._clone_module(base_module)
    assert clone is not base_module

    train._remember_compiled_module("skip", base_module, limit=0)
    assert "skip" not in train._COMPILED_MODULE_CACHE

    train._remember_compiled_module("first", base_module, limit=1)
    assert "first" in train._COMPILED_MODULE_CACHE
    train._remember_compiled_module("second", base_module, limit=1)
    assert "first" not in train._COMPILED_MODULE_CACHE

    cached = train._load_compiled_from_cache("second")
    assert cached is not None

    monkeypatch.setattr(train.torch, "compile", None)
    assert (
        train._compile_with_cache(base_module, "linear", "inductor", "default", 1)
        is base_module
    )

    compiled_calls: list[tuple[str | None, str | None]] = []

    def fake_compile(module, backend=None, mode=None):
        compiled_calls.append((backend, mode))
        return copy.deepcopy(module)

    monkeypatch.setattr(train.torch, "compile", fake_compile)
    train._COMPILED_MODULE_CACHE.clear()

    compiled_once = train._compile_with_cache(
        base_module, "linear", "fake_backend", "fake_mode", 2
    )
    assert compiled_calls == [("fake_backend", "fake_mode")]

    base_module.weight.data.fill_(3.14)
    cached_reload = train._compile_with_cache(
        base_module, "linear", "fake_backend", "fake_mode", 2
    )
    assert cached_reload is not compiled_once
    assert isinstance(cached_reload, nn.Linear)
    assert torch.allclose(cached_reload.weight, base_module.weight)


def test_layer_norm_logging_and_gradient_clipping(tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    config.model.layer_norm = True

    class PassthroughEncoder(nn.Module):
        def forward(self, inputs):
            return inputs

    feature_store: dict[str, torch.Tensor] = {}

    class RecordingFusion(nn.Module):
        def forward(self, feats, mask=None):
            feature_store["sensor"] = feats["sensor"].detach().clone()
            batch = next(iter(feats.values())).shape[0]
            num_classes = config.dataset.get("num_classes", 11)
            return torch.zeros(batch, num_classes)

    monkeypatch.setattr(
        train,
        "build_encoder",
        lambda **kwargs: PassthroughEncoder(),
    )
    monkeypatch.setattr(
        train,
        "build_fusion_model",
        lambda **kwargs: RecordingFusion(),
    )

    module = train.MultimodalFusionModule(config)
    features = {
        "sensor": torch.tensor(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float32
        )
    }
    mask = torch.ones(features["sensor"].shape[0], 1)
    logits = module.forward(features, mask)
    assert logits.shape == (features["sensor"].shape[0], 11)
    normalized = feature_store["sensor"]
    assert torch.allclose(
        normalized.mean(dim=1), torch.zeros(normalized.size(0)), atol=1e-5
    )

    logged: dict[str, tuple[torch.Tensor, dict[str, object]]] = {}

    def fake_log(self, name, value, **kwargs):
        logged[name] = (value, kwargs)

    module_any = cast(Any, module)
    module_any._trainer = object()
    module_any.log = types.MethodType(fake_log, module)
    metric_value = torch.tensor(1.23)
    module._log_metric("custom/metric", metric_value, prog_bar=True)
    assert "custom/metric" in logged
    assert logged["custom/metric"][0] is metric_value

    clip_args: dict[str, tuple[object, float, str]] = {}

    def fake_clip(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        clip_args["args"] = (optimizer, gradient_clip_val, gradient_clip_algorithm)

    module_any.clip_gradients = types.MethodType(fake_clip, module)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    module.configure_gradient_clipping(
        optimizer,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
    )
    assert clip_args["args"][1] == pytest.approx(0.5)
    assert clip_args["args"][2] == "value"


def test_maybe_compile_modules_and_forward_skip_missing(tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    config.dataset.modalities = ["sensor", "aux"]
    config.dataset.num_classes = 3
    config.model.encoders["sensor"] = {"input_dim": 64}
    config.model.encoders["aux"] = {"input_dim": 16}
    config.training.enable_compile = True
    config.training.compile_cache_size = 1

    compile_calls: list[tuple[str | None, str | None, str]] = []

    def fake_compile(module, backend=None, mode=None):
        compile_calls.append((backend, mode, module.__class__.__name__))
        return copy.deepcopy(module)

    monkeypatch.setattr(train.torch, "compile", fake_compile)

    recorded: dict[str, list[str]] = {}

    class RecorderFusion(nn.Module):
        def forward(self, feats, mask=None):
            recorded["keys"] = sorted(feats.keys())
            batch = next(iter(feats.values())).shape[0]
            return torch.zeros(batch, config.dataset.num_classes)

    monkeypatch.setattr(
        train,
        "build_fusion_model",
        lambda **_: RecorderFusion(),
    )

    module = train.MultimodalFusionModule(config)
    assert compile_calls, "_maybe_compile_modules should invoke torch.compile"

    features = {"sensor": torch.randn(5, 64)}
    mask = torch.ones(5, 2)
    logits = module(features, mask)
    assert logits.shape == (5, config.dataset.num_classes)
    assert recorded["keys"] == ["sensor"]


def test_maybe_compile_handles_missing_torch_compile(tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    config.training.enable_compile = True

    monkeypatch.setattr(train.torch, "compile", None)

    module = train.MultimodalFusionModule(config)
    assert isinstance(module.encoders["sensor"], nn.Module)


def test_train_entrypoint_executes_main(monkeypatch):
    calls: list[bool] = []

    def fake_main():
        calls.append(True)

    lines = Path("src/train.py").read_text().splitlines()
    start = next(
        idx for idx, line in enumerate(lines) if line.startswith("if __name__")
    )
    block = "\n" * start + "\n".join(lines[start:])

    namespace = {"__name__": "__main__", "main": fake_main}
    exec(compile(block, "src/train.py", "exec"), namespace)

    assert calls == [True]
