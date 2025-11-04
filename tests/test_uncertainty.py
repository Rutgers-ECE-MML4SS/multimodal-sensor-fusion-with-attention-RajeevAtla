import matplotlib
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

matplotlib.use('Agg')

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import uncertainty


class LinearModel(torch.nn.Module):
    def __init__(self, in_dim=4, num_classes=3):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, num_classes, bias=False)

    def forward(self, inputs):
        return self.linear(inputs)


def test_mc_dropout_uncertainty_restores_mode():
    base_model = LinearModel()
    base_model.eval()

    mc = uncertainty.MCDropoutUncertainty(base_model, num_samples=3)
    inputs = torch.randn(5, 4)
    mean_logits, variance = mc(inputs)

    assert mean_logits.shape == (5, 3)
    assert variance.shape == (5,)
    assert not base_model.training  # Restored eval mode


def test_calibration_metrics_and_reliability_diagram(tmp_path, monkeypatch):
    logits = torch.tensor([[2.0, 0.5], [0.2, 1.5]])
    labels = torch.tensor([0, 1])
    confidences = torch.tensor([0.8, 0.7])
    predictions = torch.tensor([0, 1])

    ece = uncertainty.CalibrationMetrics.expected_calibration_error(
        confidences, predictions, labels, num_bins=2
    )
    mce = uncertainty.CalibrationMetrics.maximum_calibration_error(
        confidences, predictions, labels, num_bins=2
    )
    nll = uncertainty.CalibrationMetrics.negative_log_likelihood(logits, labels)
    assert ece >= 0
    assert mce >= 0
    assert nll >= 0

    save_path = tmp_path / "reliability.png"
    uncertainty.CalibrationMetrics.reliability_diagram(
        confidences.numpy(),
        predictions.numpy(),
        labels.numpy(),
        num_bins=2,
        save_path=save_path,
    )
    assert save_path.exists()

    showed = {}

    def fake_show():
        showed["called"] = True

    monkeypatch.setattr("matplotlib.pyplot.show", fake_show)
    uncertainty.CalibrationMetrics.reliability_diagram(
        confidences.numpy(),
        predictions.numpy(),
        labels.numpy(),
        num_bins=2,
        save_path=None,
    )
    assert showed.get("called")


def test_uncertainty_weighted_fusion_handles_fallback():
    fusion = uncertainty.UncertaintyWeightedFusion(epsilon=1e-6)
    modality_predictions = {
        "mod1": torch.tensor([[2.0, 1.0]]),
        "mod2": torch.tensor([[1.0, 3.0]]),
    }
    modality_uncertainties = {
        "mod1": torch.tensor([0.1]),
        "mod2": torch.tensor([0.2]),
    }
    mask = torch.tensor([[1.0, 1.0]])
    fused, weights = fusion(modality_predictions, modality_uncertainties, mask)
    assert fused.shape == (1, 2)
    assert weights.sum().item() == pytest.approx(1.0)

    # Fallback branch when no modalities available
    zero_mask = torch.zeros_like(mask)
    fused, weights = fusion(modality_predictions, modality_uncertainties, zero_mask)
    assert torch.allclose(weights, torch.full_like(weights, 0.5))


def test_temperature_scaling_and_calibration():
    model = uncertainty.TemperatureScaling()
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.5]])
    scaled = model(logits)
    assert scaled.shape == logits.shape

    labels = torch.tensor([0, 1])
    model.temperature = torch.nn.Parameter(torch.ones(1, device="meta"))
    model.calibrate(logits, labels, lr=0.1, max_iter=5)
    assert model.temperature.item() > 0


def test_temperature_scaling_device_mismatch(monkeypatch):
    model = uncertainty.TemperatureScaling()
    base_logits = torch.randn(4, 3)

    class FakeCudaTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, tensor):
            return torch.Tensor._make_subclass(cls, tensor, tensor.requires_grad)

        @property
        def device(self):
            return torch.device("cuda")

    fake_logits = FakeCudaTensor(base_logits)
    labels = torch.randint(0, 3, (4,))

    calls: dict[str, bool] = {}
    original_to = torch.Tensor.to

    def fake_to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get("device")
        if isinstance(device, torch.device) and device.type == "cuda":
            calls["cuda"] = True
            return original_to(self, torch.device("cpu"))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    model.temperature = torch.nn.Parameter(torch.ones(1))
    model.calibrate(fake_logits, labels, lr=0.05, max_iter=2)

    assert calls.get("cuda") is True
    assert model.temperature.device.type == "cpu"


def test_ensemble_uncertainty_and_empty_guard():
    ensemble_empty = uncertainty.EnsembleUncertainty([])
    with pytest.raises(ValueError):
        ensemble_empty.predict_with_uncertainty(torch.randn(1, 4))
    inputs = torch.randn(4, 4)
    models = [LinearModel(), LinearModel()]
    ensemble = uncertainty.EnsembleUncertainty(models)
    mean_probs, uncert = ensemble.predict_with_uncertainty(inputs)
    assert mean_probs.shape[0] == inputs.shape[0]
    assert uncert.shape[0] == inputs.shape[0]


class SimpleDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        inputs = torch.randn(4)
        label = torch.tensor(idx % 2, dtype=torch.long)
        return inputs, label


def test_compute_calibration_metrics():
    model = LinearModel()
    dataset = SimpleDataset()
    loader = DataLoader(dataset, batch_size=2)
    metrics = uncertainty.compute_calibration_metrics(model, loader)

    assert set(metrics.keys()) == {"ece", "mce", "nll", "accuracy"}


def test_uncertainty_weighted_fusion_errors():
    fusion = uncertainty.UncertaintyWeightedFusion()
    with pytest.raises(ValueError):
        fusion({}, {}, torch.ones(1, 1))
    with pytest.raises(KeyError):
        fusion({"mod1": torch.zeros(1, 2)}, {}, torch.ones(1, 1))


def test_uncertainty_main_generates_outputs(capsys, tmp_path):
    output_path = tmp_path / "cli_reliability.png"
    results = uncertainty.main(save_path=output_path, num_samples=32, num_classes=4)
    captured = capsys.readouterr().out
    assert "Testing calibration metrics..." in captured
    assert "Reliability diagram created" in captured
    assert output_path.exists()
    assert results["diagram_created"] is True
    assert results["save_path"] == str(output_path)
    assert isinstance(results["ece"], float)


def test_uncertainty_main_handles_not_implemented(monkeypatch, capsys, tmp_path):
    def raise_ece(*args, **kwargs):
        raise NotImplementedError("ece unavailable")

    def raise_reliability(*args, **kwargs):
        raise NotImplementedError("reliability unavailable")

    monkeypatch.setattr(
        uncertainty.CalibrationMetrics,
        "expected_calibration_error",
        staticmethod(raise_ece),
    )
    monkeypatch.setattr(
        uncertainty.CalibrationMetrics,
        "reliability_diagram",
        staticmethod(raise_reliability),
    )

    results = uncertainty.main(save_path=tmp_path / "unused.png", num_samples=8, num_classes=2)
    captured = capsys.readouterr().out
    assert "ECE not implemented yet" in captured
    assert "Reliability diagram not implemented yet" in captured
    assert results["ece"] is None
    assert results["diagram_created"] is False


def test_uncertainty_module_entrypoint(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    runpy.run_module("uncertainty", run_name="__main__")
    captured = capsys.readouterr().out
    assert "Testing calibration metrics..." in captured
    assert "Reliability diagram created" in captured
    assert (tmp_path / "test_reliability.png").exists()
