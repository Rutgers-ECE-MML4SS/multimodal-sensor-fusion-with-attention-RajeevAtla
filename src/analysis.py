"""
Analysis and Visualization Script

Generates all required plots for experiments:
- Fusion strategy comparison
- Missing modality robustness
- Attention visualization
- Calibration reliability diagrams
"""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_fusion_comparison(
    results: Mapping[str, Any],
    save_path: Path | str = Path("analysis/fusion_comparison.png"),
) -> None:
    """
    Plot comparison of fusion strategies.

    Args:
        results: Dict from experiments/fusion_comparison.json
        save_path: Where to save the plot
    """
    # Extract data
    strategies = list(results["results"].keys())
    accuracies = [results["results"][s]["accuracy"] for s in strategies]
    f1_scores = [results["results"][s]["f1_macro"] for s in strategies]
    eces = [results["results"][s]["ece"] for s in strategies]
    inference_times = [
        results["results"][s]["inference_ms"] for s in strategies
    ]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Fusion Strategy Comparison", fontsize=16, fontweight="bold")

    # Accuracy comparison
    axes[0, 0].bar(
        strategies, accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )
    axes[0, 0].set_ylabel("Accuracy", fontsize=12)
    axes[0, 0].set_title("Test Accuracy", fontsize=12)
    axes[0, 0].set_ylim([0, 1.0])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    # F1 score comparison
    axes[0, 1].bar(
        strategies, f1_scores, color=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )
    axes[0, 1].set_ylabel("F1 Score (macro)", fontsize=12)
    axes[0, 1].set_title("F1 Score", fontsize=12)
    axes[0, 1].set_ylim([0, 1.0])
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    # ECE comparison
    axes[1, 0].bar(strategies, eces, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1, 0].set_ylabel("ECE", fontsize=12)
    axes[1, 0].set_title("Expected Calibration Error", fontsize=12)
    axes[1, 0].axhline(y=0.1, color="r", linestyle="--", label="Target (0.1)")
    axes[1, 0].legend()
    for i, v in enumerate(eces):
        axes[1, 0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)

    # Inference time comparison
    axes[1, 1].bar(
        strategies, inference_times, color=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )
    axes[1, 1].set_ylabel("Inference Time (ms)", fontsize=12)
    axes[1, 1].set_title("Inference Speed", fontsize=12)
    for i, v in enumerate(inference_times):
        axes[1, 1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    plt.tight_layout()

    # Save
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Fusion comparison plot saved to: {output_path}")
    plt.close()


def plot_missing_modality_robustness(
    results: Mapping[str, Any],
    save_path: Path | str = Path("analysis/missing_modality.png"),
) -> None:
    """
    Plot performance degradation with missing modalities.

    Args:
        results: Dict from experiments/missing_modality.json
        save_path: Where to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Missing Modality Robustness", fontsize=16, fontweight="bold")

    # Left plot: Performance vs number of modalities
    all_combos = results["all_combinations"]
    modality_counts = {}

    for combo_name, metrics in all_combos.items():
        num_modalities = len(combo_name.split("+"))
        if num_modalities not in modality_counts:
            modality_counts[num_modalities] = []
        modality_counts[num_modalities].append(metrics["accuracy"])

    # Compute mean and std for each count
    counts = sorted(modality_counts.keys())
    means = [np.mean(modality_counts[c]) for c in counts]
    stds = [np.std(modality_counts[c]) for c in counts]

    axes[0].errorbar(
        counts,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
        label="Accuracy",
    )
    axes[0].fill_between(
        counts,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.3,
    )
    axes[0].set_xlabel("Number of Available Modalities", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Performance vs Modality Availability", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(counts)
    axes[0].set_ylim([0, 1.0])

    # Right plot: Single modality comparison
    single_modalities = results["single_modalities"]
    modality_names = list(single_modalities.keys())
    accuracies = [single_modalities[m]["accuracy"] for m in modality_names]

    colors = plt.get_cmap("Set3")(np.linspace(0, 1, len(modality_names)))
    axes[1].bar(range(len(modality_names)), accuracies, color=colors)
    axes[1].set_xticks(range(len(modality_names)))
    axes[1].set_xticklabels(modality_names, rotation=45, ha="right")
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Single Modality Performance", fontsize=12)
    axes[1].set_ylim([0, 1.0])
    axes[1].axhline(
        y=results["full_modalities"]["accuracy"],
        color="r",
        linestyle="--",
        label="Full (all modalities)",
    )
    axes[1].legend()

    # Add value labels
    for i, v in enumerate(accuracies):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()

    # Save
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Missing modality plot saved to: {output_path}")
    plt.close()


def plot_attention_weights(
    attention_weights: np.ndarray,
    modality_names: Sequence[str],
    save_path: Path | str = Path("analysis/attention_viz.png"),
) -> None:
    """
    Visualize attention weights between modalities.

    Args:
        attention_weights: (num_modalities, num_modalities) attention matrix
        modality_names: List of modality names
        save_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(attention_weights, cmap="YlOrRd", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(len(modality_names)))
    ax.set_yticks(range(len(modality_names)))
    ax.set_xticklabels(modality_names, rotation=45, ha="right")
    ax.set_yticklabels(modality_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight", rotation=270, labelpad=20, fontsize=12)

    # Add text annotations
    for i in range(len(modality_names)):
        for j in range(len(modality_names)):
            ax.text(
                j,
                i,
                f"{attention_weights[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    ax.set_title(
        "Cross-Modal Attention Weights", fontsize=14, fontweight="bold", pad=20
    )
    ax.set_xlabel("Attended Modality (Key)", fontsize=12)
    ax.set_ylabel("Query Modality", fontsize=12)

    plt.tight_layout()

    # Save
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Attention visualization saved to: {output_path}")
    plt.close()


def plot_calibration_diagram(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
    save_path: Path | str = Path("analysis/calibration.png"),
) -> None:
    """
    Plot reliability diagram for calibration.

    Args:
        confidences: (N,) predicted confidence scores
        predictions: (N,) predicted class indices
        labels: (N,) ground truth labels
        num_bins: Number of confidence bins
        save_path: Where to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Calibration Analysis", fontsize=16, fontweight="bold")

    # Compute bin statistics
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(num_bins):
        bin_mask = (confidences >= bin_edges[i]) & (
            confidences < bin_edges[i + 1]
        )
        if i == num_bins - 1:  # Include right edge in last bin
            bin_mask = (confidences >= bin_edges[i]) & (
                confidences <= bin_edges[i + 1]
            )

        if bin_mask.sum() > 0:
            bin_conf = confidences[bin_mask].mean()
            bin_acc = (predictions[bin_mask] == labels[bin_mask]).mean()
            bin_confidences.append(bin_conf)
            bin_accuracies.append(bin_acc)
            bin_counts.append(bin_mask.sum())
        else:
            bin_confidences.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accuracies.append(0)
            bin_counts.append(0)

    # Left plot: Reliability diagram
    axes[0].bar(
        range(num_bins),
        bin_accuracies,
        width=0.8,
        alpha=0.7,
        label="Accuracy",
        edgecolor="black",
    )
    axes[0].plot(
        range(num_bins),
        bin_confidences,
        "ro-",
        linewidth=2,
        markersize=6,
        label="Confidence",
    )
    axes[0].plot(
        [0, num_bins - 1],
        [bin_accuracies[0], bin_accuracies[-1]],
        "k--",
        alpha=0.5,
        label="Perfect Calibration",
    )
    axes[0].set_xlabel("Confidence Bin", fontsize=12)
    axes[0].set_ylabel("Accuracy / Confidence", fontsize=12)
    axes[0].set_title("Reliability Diagram", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.0])

    # Right plot: Confidence histogram
    axes[1].hist(confidences, bins=50, alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Confidence Score", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Confidence Distribution", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Compute and display ECE
    ece = sum(
        [
            abs(bin_accuracies[i] - bin_confidences[i]) * bin_counts[i]
            for i in range(num_bins)
        ]
    ) / sum(bin_counts)
    axes[0].text(
        0.05,
        0.95,
        f"ECE: {ece:.4f}",
        transform=axes[0].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Calibration diagram saved to: {output_path}")
    plt.close()


def generate_all_plots(
    experiment_dir: Path | str, output_dir: Path | str
) -> None:
    """
    Generate plots for each fusion type under experiments/<fusion>/.
    """
    experiment_path = Path(experiment_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating Analysis Plots")
    print("=" * 80)

    subdirs = sorted(
        d for d in experiment_path.iterdir() if d.is_dir() and d.name
    )

    if not subdirs:
        print(f"No subdirectories found in {experiment_path}, nothing to plot.")
        return

    for idx, subdir in enumerate(subdirs, start=1):
        fusion_name = subdir.name
        print(f"\n[{idx}/{len(subdirs)}] Fusion type: {fusion_name}")
        fusion_output = output_path / fusion_name
        fusion_output.mkdir(parents=True, exist_ok=True)

        fusion_file = subdir / "fusion_comparison.json"
        if fusion_file.exists():
            print("  Generating fusion comparison plot...")
            with open(fusion_file) as f:
                results = json.load(f)
            plot_fusion_comparison(
                results, fusion_output / "fusion_comparison.png"
            )
        else:
            print(
                f"  Warning: {fusion_file} not found. Skipping fusion comparison."
            )

        missing_file = subdir / "missing_modality.json"
        if missing_file.exists():
            print("  Generating missing modality plot...")
            with open(missing_file) as f:
                results = json.load(f)
            plot_missing_modality_robustness(
                results, fusion_output / "missing_modality.png"
            )
        else:
            print(
                f"  Warning: {missing_file} not found. Skipping missing modality plot."
            )

    print("\n" + "=" * 80)
    print("Plot generation complete!")
    print(f"Plots saved to: {output_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate analysis plots")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiments",
        help="Directory with experiment JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis",
        help="Directory to save plots",
    )

    args = parser.parse_args()

    generate_all_plots(args.experiment_dir, args.output_dir)


if __name__ == "__main__":
    main()
