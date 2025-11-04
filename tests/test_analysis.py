import json
import runpy
import sys
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

matplotlib.use("Agg")  # Use non-interactive backend for tests

import analysis


def _fusion_results():
    return {
        "results": {
            "early": {
                "accuracy": 0.8,
                "f1_macro": 0.75,
                "ece": 0.05,
                "inference_ms": 5.0,
            },
            "late": {
                "accuracy": 0.77,
                "f1_macro": 0.7,
                "ece": 0.08,
                "inference_ms": 4.5,
            },
            "hybrid": {
                "accuracy": 0.82,
                "f1_macro": 0.78,
                "ece": 0.04,
                "inference_ms": 6.0,
            },
        }
    }


def _missing_results():
    return {
        "full_modalities": {"accuracy": 0.88},
        "single_modalities": {
            "video": {"accuracy": 0.75},
            "imu": {"accuracy": 0.7},
            "audio": {"accuracy": 0.68},
        },
        "all_combinations": {
            "video+imu": {"accuracy": 0.83},
            "video+audio": {"accuracy": 0.81},
            "imu+audio": {"accuracy": 0.79},
            "video": {"accuracy": 0.75},
            "imu": {"accuracy": 0.7},
            "audio": {"accuracy": 0.68},
        },
    }


def test_plotting_functions_create_outputs(tmp_path):
    fusion_path = tmp_path / "fusion.png"
    missing_path = tmp_path / "missing.png"
    attention_path = tmp_path / "attention.png"
    calibration_path = tmp_path / "calibration.png"
    empty_bin_path = tmp_path / "empty_calibration.png"

    analysis.plot_fusion_comparison(_fusion_results(), fusion_path)
    analysis.plot_missing_modality_robustness(_missing_results(), missing_path)

    weights = np.array([[0.4, 0.6], [0.3, 0.7]])
    analysis.plot_attention_weights(weights, ["video", "imu"], attention_path)

    confidences = np.linspace(0.1, 0.9, 9)
    predictions = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    labels = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0])

    analysis.plot_calibration_diagram(
        confidences, predictions, labels, num_bins=3, save_path=calibration_path
    )
    analysis.plot_calibration_diagram(
        np.array([0.1, 0.9]),
        np.array([0, 1]),
        np.array([0, 1]),
        num_bins=4,
        save_path=empty_bin_path,
    )

    for path in [fusion_path, missing_path, attention_path, calibration_path, empty_bin_path]:
        assert path.exists()
        assert path.stat().st_size > 0


def test_generate_all_plots_handles_missing_files(tmp_path, capsys):
    # Directory with both JSON files should trigger plotting
    experiment_with_files = tmp_path / "exp_with_files"
    experiment_with_files.mkdir()
    (experiment_with_files / "fusion_comparison.json").write_text(
        json.dumps(_fusion_results())
    )
    (experiment_with_files / "missing_modality.json").write_text(
        json.dumps(_missing_results())
    )

    output_dir = tmp_path / "analysis_outputs"
    analysis.generate_all_plots(experiment_with_files, output_dir)

    assert (output_dir / "fusion_comparison.png").exists()
    assert (output_dir / "missing_modality.png").exists()

    # Directory without JSON files exercises warning branches
    experiment_without_files = tmp_path / "exp_without_files"
    experiment_without_files.mkdir()

    analysis.generate_all_plots(experiment_without_files, output_dir)
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_analysis_main_cli(tmp_path, monkeypatch):
    experiment_dir = tmp_path / "cli_exp"
    experiment_dir.mkdir()
    (experiment_dir / "fusion_comparison.json").write_text(json.dumps(_fusion_results()))
    (experiment_dir / "missing_modality.json").write_text(json.dumps(_missing_results()))

    output_dir = tmp_path / "cli_output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analysis.py",
            "--experiment_dir",
            str(experiment_dir),
            "--output_dir",
            str(output_dir),
        ],
    )
    runpy.run_module("analysis", run_name="__main__")

    assert (output_dir / "fusion_comparison.png").exists()
    assert (output_dir / "missing_modality.png").exists()
