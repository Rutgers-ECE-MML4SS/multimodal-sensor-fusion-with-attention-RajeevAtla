import json
import sys
import runpy
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import eval as evaluation


class DummyModel(torch.nn.Module):
    def __init__(self, num_modalities=2, num_classes=3):
        super().__init__()
        object.__setattr__(self, "num_modalities", num_modalities)
        object.__setattr__(self, "num_classes", num_classes)

    def forward(self, features, mask):
        summed = torch.zeros(next(iter(features.values())).shape[0])
        for idx, tensor in enumerate(features.values()):
            weight = mask[:, idx] if mask is not None else torch.ones_like(summed)
            summed = summed + tensor.sum(dim=1) * weight

        logits = torch.stack(
            [summed, summed * 0.1, -summed],
            dim=1,
        )
        return logits


def _make_batch(batch_size=2, num_modalities=2):
    features = {
        f"mod{i}": torch.ones(batch_size, 4) * (i + 1) for i in range(num_modalities)
    }
    labels = torch.tensor(list(range(batch_size))) % 3
    mask = torch.ones(batch_size, num_modalities)
    return features, labels, mask


def test_evaluate_model_and_predictions(monkeypatch):
    dataloader = [_make_batch()]
    monkeypatch.setattr(evaluation, "tqdm", lambda iterable, **_: iterable)

    model = DummyModel()
    metrics, predictions = evaluation.evaluate_model(
        model, dataloader, device="cpu", return_predictions=True
    )

    preds, labels, confidences = predictions
    assert metrics["num_samples"] == len(labels)
    assert preds.shape == labels.shape
    assert confidences.shape == labels.shape


def test_evaluate_missing_modalities_computes_importance():
    dataloader = [_make_batch()]
    model = DummyModel()

    results = evaluation.evaluate_missing_modalities(
        model, dataloader, modality_names=["mod0", "mod1"], device="cpu"
    )

    assert "full_modalities" in results
    assert "single_modalities" in results
    assert "modality_importance" in results
    # Importance scores should be present for each modality
    assert set(results["modality_importance"].keys()) == {"mod0", "mod1"}


def test_compute_modality_importance_balanced():
    results = {
        "all_combinations": {
            "A+B": {"accuracy": 0.9},
            "A": {"accuracy": 0.8},
            "B": {"accuracy": 0.7},
        }
    }
    importance = evaluation._compute_modality_importance(results, ["A", "B"])
    assert set(importance.keys()) == {"A", "B"}


def test_save_results_json(tmp_path):
    output = tmp_path / "results.json"
    evaluation.save_results_json({"metric": 0.5}, output)
    data = output.read_text()
    assert '"metric": 0.5' in data


def test_evaluate_with_modality_subset():
    dataloader = [_make_batch()]
    model = DummyModel()
    metrics = evaluation._evaluate_with_modality_subset(
        model=model,
        dataloader=dataloader,
        available_indices=[0],
        total_modalities=2,
        device="cpu",
    )
    assert "accuracy" in metrics and "f1_macro" in metrics


def test_evaluate_model_without_predictions(monkeypatch):
    dataloader = [_make_batch()]
    monkeypatch.setattr(evaluation, "tqdm", lambda iterable, **_: iterable)
    model = DummyModel()
    metrics = evaluation.evaluate_model(model, dataloader, device="cpu")
    assert "accuracy" in metrics


def test_compute_modality_importance_handles_missing_data():
    results = {"all_combinations": {"A": {"accuracy": 0.5}}}
    importance = evaluation._compute_modality_importance(results, ["A", "B"])
    assert importance["B"] == 0.0


def _configure_cli_mocks(monkeypatch, tmp_path, missing: bool = False):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("dummy")
    output_dir = tmp_path / "outputs"

    class DummyLightningModel:
        def __init__(self):
            dataset_cfg = type(
                "DatasetCfg",
                (),
                {
                    "name": "synthetic",
                    "data_dir": "",
                    "modalities": ["mod0"],
                    "batch_size": 1,
                    "num_workers": 0,
                },
            )()
            model_cfg = type("ModelCfg", (), {"fusion_type": "hybrid"})()
            self.config = type("Config", (), {"dataset": dataset_cfg, "model": model_cfg})()

        def eval(self):
            return self

        def to(self, device):
            return self

    class LoaderWrapper:
        @staticmethod
        def load_from_checkpoint(path):
            assert Path(path) == checkpoint
            return DummyLightningModel()

    metrics = {"accuracy": 0.9, "f1_macro": 0.8, "loss": 0.2, "num_samples": 2}
    predictions = (
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.tensor([0.6, 0.7]),
    )

    monkeypatch.setattr(evaluation, "MultimodalFusionModule", LoaderWrapper)
    monkeypatch.setattr(
        evaluation,
        "create_dataloaders",
        lambda **_: (["train"], ["val"], ["test_loader"]),
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_model",
        lambda *args, **kwargs: (metrics, predictions),
    )

    missing_results = {
        "full_modalities": {"accuracy": 0.85},
        "single_modalities": {"mod0": {"accuracy": 0.8}},
        "all_combinations": {"mod0": {"accuracy": 0.8}},
        "modality_importance": {"mod0": 1.0},
    }

    evaluate_missing_spy = {"called": False}

    def fake_missing(*args, **kwargs):
        evaluate_missing_spy["called"] = True
        return missing_results

    monkeypatch.setattr(
        evaluation,
        "evaluate_missing_modalities",
        fake_missing if missing else lambda *a, **k: pytest.fail(
            "Missing modality branch should not execute without flag"
        ),
    )

    class DummyCalibration:
        @staticmethod
        def expected_calibration_error(*args, **kwargs):
            return 0.05

    monkeypatch.setattr(evaluation, "CalibrationMetrics", DummyCalibration)

    argv = [
        "eval.py",
        "--checkpoint",
        str(checkpoint),
        "--output_dir",
        str(output_dir),
    ]
    if missing:
        argv.append("--missing_modality_test")
    monkeypatch.setattr(sys, "argv", argv)

    return output_dir, missing_results, evaluate_missing_spy


def test_eval_main_cli_standard(tmp_path, monkeypatch, capsys):
    output_dir, _, evaluate_missing_spy = _configure_cli_mocks(monkeypatch, tmp_path)
    evaluation.main()

    captured = capsys.readouterr().out
    assert "Standard Evaluation" in captured
    assert "ECE" in captured
    assert not evaluate_missing_spy["called"]

    results_path = output_dir / "evaluation_results.json"
    assert results_path.exists()
    saved = json.loads(results_path.read_text())
    assert saved["test_accuracy"] == 0.9


def test_eval_main_cli_missing_modalities(tmp_path, monkeypatch, capsys):
    output_dir, missing_results, evaluate_missing_spy = _configure_cli_mocks(
        monkeypatch, tmp_path, missing=True
    )
    evaluation.main()

    captured = capsys.readouterr().out
    assert "Missing Modality Robustness Test" in captured
    assert evaluate_missing_spy["called"]
    assert "Summary" in captured

    missing_path = output_dir / "missing_modality.json"
    standard_path = output_dir / "evaluation_results.json"
    assert missing_path.exists()
    assert standard_path.exists()
    saved_missing = json.loads(missing_path.read_text())
    assert saved_missing == missing_results


def test_eval_script_entrypoint_runs(tmp_path, monkeypatch, capsys):
    import sys as system_mod

    checkpoint = tmp_path / "script.ckpt"
    checkpoint.write_text("dummy")
    output_dir = tmp_path / "script_output"

    class LightningWrapper(DummyModel):
        def __init__(self):
            super().__init__(num_modalities=1, num_classes=3)
            dataset_cfg = type(
                "DatasetCfg",
                (),
                {
                    "name": "synthetic",
                    "data_dir": "",
                    "modalities": ["mod0"],
                    "batch_size": 2,
                    "num_workers": 0,
                },
            )()
            model_cfg = type("ModelCfg", (), {"fusion_type": "hybrid"})()
            self.config = type("Config", (), {"dataset": dataset_cfg, "model": model_cfg})()

    class LoaderWrapper:
        @staticmethod
        def load_from_checkpoint(path):
            assert Path(path) == checkpoint
            return LightningWrapper()

    def dataloader_factory(**kwargs):
        batch = _make_batch(batch_size=2, num_modalities=1)
        loader = [batch]
        return loader, loader, loader

    class CalibrationWrapper:
        @staticmethod
        def expected_calibration_error(*args, **kwargs):
            return 0.02

    tqdm_stub = types.ModuleType("tqdm")

    def passthrough(iterable, **kwargs):
        return iterable

    setattr(tqdm_stub, "tqdm", passthrough)

    monkeypatch.setitem(system_mod.modules, "tqdm", tqdm_stub)
    monkeypatch.setitem(system_mod.modules, "train", types.ModuleType("train"))
    monkeypatch.setitem(system_mod.modules, "data", types.ModuleType("data"))
    monkeypatch.setitem(system_mod.modules, "uncertainty", types.ModuleType("uncertainty"))

    setattr(system_mod.modules["train"], "MultimodalFusionModule", LoaderWrapper)
    setattr(system_mod.modules["data"], "create_dataloaders", dataloader_factory)
    setattr(system_mod.modules["uncertainty"], "CalibrationMetrics", CalibrationWrapper)

    argv = [
        "eval.py",
        "--checkpoint",
        str(checkpoint),
        "--output_dir",
        str(output_dir),
    ]
    monkeypatch.setattr(system_mod, "argv", argv)

    existing_eval = system_mod.modules.pop("eval", None)
    try:
        runpy.run_module("eval", run_name="__main__")
    finally:
        if existing_eval is not None:
            system_mod.modules["eval"] = existing_eval

    captured = capsys.readouterr().out
    assert "Standard Evaluation" in captured
    assert (output_dir / "evaluation_results.json").exists()
