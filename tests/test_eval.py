import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import eval as evaluation


class DummyModel(torch.nn.Module):
    def __init__(self, num_modalities=2, num_classes=3):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes = num_classes

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


def test_eval_main_cli(tmp_path, capsys):
    import argparse
    import json as json_mod
    import numpy as np

    metrics = {"accuracy": 1.0, "f1_macro": 1.0, "loss": 0.0, "num_samples": 1}
    predictions = (
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        torch.ones(1),
    )

    class DummyModel:
        def __init__(self):
            dataset = type(
                "DatasetCfg",
                (),
                {
                    "name": "synthetic",
                    "data_dir": "",
                    "modalities": ["mod0"],
                    "batch_size": 1,
                    "num_workers": 0,
                },
            )
            model = type("ModelCfg", (), {"fusion_type": "hybrid"})
            self.config = type("Config", (), {"dataset": dataset, "model": model})

        def eval(self):
            return self

        def to(self, device):
            return self

    globals_dict = {
        "__name__": "__main__",
        "__file__": str(Path("src/eval.py")),
        "torch": torch,
        "np": np,
        "F": torch.nn.functional,
        "json": json_mod,
        "Path": Path,
        "argparse": argparse,
        "tqdm": lambda iterable, **kwargs: iterable,
        "MultimodalFusionModule": type(
            "Loader", (), {"load_from_checkpoint": staticmethod(lambda _path: DummyModel())}
        ),
        "create_dataloaders": lambda **kwargs: ([None], [None], [None]),
        "CalibrationMetrics": type(
            "Cal", (), {"expected_calibration_error": staticmethod(lambda *args, **kwargs: 0.0)}
        ),
        "evaluate_model": lambda *args, **kwargs: (metrics, predictions),
        "evaluate_missing_modalities": lambda *args, **kwargs: {
            "full_modalities": {"accuracy": 1.0},
            "single_modalities": {"mod0": {"accuracy": 1.0}},
            "all_combinations": {"mod0": {"accuracy": 1.0}},
            "modality_importance": {"mod0": 1.0},
        },
        "save_results_json": lambda results, path: Path(path).write_text(
            json_mod.dumps(results)
        ),
    }
    exec(Path("src/eval.py").read_text(), globals_dict)
    output = capsys.readouterr().out
    assert "Evaluation complete!" in output
