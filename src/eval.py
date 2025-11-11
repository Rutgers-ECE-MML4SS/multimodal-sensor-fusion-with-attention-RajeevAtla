"""
Evaluation Script for Multimodal Sensor Fusion

Provides framework for:
- Standard evaluation on test set
- Missing modality robustness testing
- Generating results for experiments/ directory
"""

import argparse
import itertools
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

from data import create_dataloaders
from train import MultimodalFusionModule
from uncertainty import CalibrationMetrics


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Retrieve config values from DictConfig, dict, or objects."""

    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def evaluate_model(
    model,
    dataloader,
    device="cpu",
    return_predictions: bool = False,
    include_logits: bool = False,
):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        return_predictions: If True, return tensors for preds/labels/confidences
        include_logits: When returning predictions, also include raw logits

    Returns:
        metrics: Dict with accuracy, loss, etc.
        predictions: Optional tuple of (preds, labels, confidences[, logits])
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_confidences = []
    all_logits = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            features, labels, mask = batch

            # Move to device
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            mask = mask.to(device)

            # Forward pass
            logits = model(features, mask)
            all_logits.append(logits.cpu())

            # Compute loss
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            # Get predictions and confidences
            probs = F.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_confidences.append(confidences.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_confidences = torch.cat(all_confidences)
    all_logits = torch.cat(all_logits)

    # Compute metrics
    accuracy = (all_preds == all_labels).float().mean().item()
    avg_loss = total_loss / num_batches

    # Compute F1 score (macro)
    from sklearn.metrics import f1_score

    f1_macro = f1_score(
        all_labels.numpy(), all_preds.numpy(), average="macro", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "loss": avg_loss,
        "num_samples": len(all_labels),
    }

    if return_predictions:
        prediction_tuple: tuple[torch.Tensor, ...] = (
            all_preds,
            all_labels,
            all_confidences,
        )
        if include_logits:
            prediction_tuple = (*prediction_tuple, all_logits)
        return metrics, prediction_tuple
    else:
        return metrics


def _parse_latency_batch(
    batch: Any,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None] | None:
    """Best-effort parsing of batches for latency measurement."""
    if isinstance(batch, Mapping):
        features = dict(batch)
        return features, None, None

    if isinstance(batch, Sequence) and batch:
        first = batch[0]
        if isinstance(first, Mapping):
            features = dict(first)
            labels = batch[1] if len(batch) > 1 else None
            mask = batch[2] if len(batch) > 2 else None
            return features, labels, mask

    return None


def _infer_batch_size(
    labels: Any, features: Mapping[str, torch.Tensor]
) -> int | None:
    """Infer batch size from labels or feature tensors."""
    if hasattr(labels, "shape"):
        return int(labels.shape[0])

    for tensor in features.values():
        if hasattr(tensor, "shape"):
            return int(tensor.shape[0])

    return None


def measure_inference_latency(model, dataloader, device="cpu") -> tuple[float, float]:
    """
    Measure per-sample inference latency statistics (mean/std in ms).

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on

    Returns:
        mean_ms: Average per-sample latency (ms)
        std_ms: Standard deviation of per-sample latency (ms)
    """
    model.eval()
    model.to(device)
    per_sample_ms: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            parsed = _parse_latency_batch(batch)
            if parsed is None:
                print("  Warning: Unable to parse batch for latency measurement, skipping.")
                continue

            features, labels, mask = parsed
            batch_size = _infer_batch_size(labels, features)
            if batch_size is None or batch_size == 0:
                print("  Warning: Unable to infer batch size for latency measurement, skipping.")
                continue

            if not features:
                print("  Warning: Empty feature dict encountered during latency measurement, skipping.")
                continue

            try:
                features = {k: v.to(device) for k, v in features.items()}
            except AttributeError:
                print("  Warning: Non-tensor feature encountered, skipping batch for latency measurement.")
                continue

            if mask is None:
                mask = torch.ones(
                    batch_size,
                    max(1, len(features)),
                    device=device,
                    dtype=next(iter(features.values())).dtype,
                )
            else:
                mask = mask.to(device)

            batch_start = time.perf_counter()
            try:
                model(features, mask)
            except TypeError:
                print("  Warning: Model call failed during latency measurement, skipping batch.")
                continue
            elapsed = time.perf_counter() - batch_start
            per_sample_ms.append((elapsed / batch_size) * 1000.0)

    if not per_sample_ms:
        return 0.0, 0.0
    arr = np.asarray(per_sample_ms, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def generate_attention_visualization(
    model: MultimodalFusionModule,
    dataloader: torch.utils.data.DataLoader,
    modality_names: Sequence[str],
    save_path: Path,
    device: str,
) -> Path | None:
    """
    Generate an attention heatmap for the hybrid fusion model.
    """
    if not modality_names:
        return None

    if model.config.model.fusion_type != "hybrid":
        return None

    data_iter = iter(dataloader)
    try:
        features, _, mask = next(data_iter)
    except StopIteration:
        return None

    features = {k: v.to(device) for k, v in features.items()}
    mask = mask.to(device)

    with torch.no_grad():
        try:
            _, attention_info = model(features, mask, return_attention=True)
        except (ValueError, TypeError):
            return None

    attention_maps = attention_info.get("attention_maps", {})
    if not attention_maps:
        return None

    num_modalities = len(modality_names)
    matrix = np.zeros((num_modalities, num_modalities), dtype=np.float32)
    counts = np.zeros_like(matrix)

    for key, weights in attention_maps.items():
        if "_to_" not in key:
            continue
        query_mod, key_mod = key.split("_to_", 1)
        if (
            query_mod not in modality_names
            or key_mod not in modality_names
        ):
            continue
        q_idx = modality_names.index(query_mod)
        k_idx = modality_names.index(key_mod)
        scalar = weights.detach().float().mean().item()
        matrix[q_idx, k_idx] += scalar
        counts[q_idx, k_idx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.divide(
            matrix,
            np.where(counts == 0, 1.0, counts),
            out=np.zeros_like(matrix),
            where=counts != 0,
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="magma", aspect="equal")
    ax.set_xticks(range(num_modalities))
    ax.set_yticks(range(num_modalities))
    ax.set_xticklabels(modality_names, rotation=45, ha="right")
    ax.set_yticklabels(modality_names)
    ax.set_xlabel("Key Modality")
    ax.set_ylabel("Query Modality")
    ax.set_title("Cross-Modal Attention Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def evaluate_missing_modalities(
    model, dataloader, modality_names, device="cpu"
):
    """
    Test model robustness to missing modalities.

    Tests all possible subsets of modalities (2^M - 1 combinations).

    Args:
        model: Trained model
        dataloader: Data loader
        modality_names: List of modality names
        device: Device to run on

    Returns:
        results: Dict with performance for each modality combination
    """
    model.eval()
    model.to(device)

    num_modalities = len(modality_names)
    results = {
        "full_modalities": {},
        "single_modalities": {},
        "all_combinations": {},
    }

    # Test all combinations
    print("\nTesting missing modality robustness...")

    for num_available in range(1, num_modalities + 1):
        print(f"\n{num_available}/{num_modalities} modalities available:")

        # Generate all combinations of this size
        for modality_indices in itertools.combinations(
            range(num_modalities), num_available
        ):
            modality_subset = [modality_names[i] for i in modality_indices]
            subset_name = "+".join(modality_subset)

            print(f"  Testing: {subset_name}")

            # Evaluate with this modality subset
            metrics = _evaluate_with_modality_subset(
                model, dataloader, modality_indices, num_modalities, device
            )

            results["all_combinations"][subset_name] = metrics

            # Store single modality results separately
            if num_available == 1:
                results["single_modalities"][modality_subset[0]] = metrics

            # Store full modality results
            if num_available == num_modalities:
                results["full_modalities"] = metrics

    # Compute modality importance (contribution when added)
    results["modality_importance"] = _compute_modality_importance(
        results, modality_names
    )

    return results


def _evaluate_with_modality_subset(
    model, dataloader, available_indices, total_modalities, device
):
    """Evaluate model with specific subset of modalities available."""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features, labels, mask = batch

            # Move to device
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)

            # Create mask for available modalities
            batch_size = labels.size(0)
            mask = torch.zeros(batch_size, total_modalities, device=device)
            for idx in available_indices:
                mask[:, idx] = 1

            # Zero out unavailable modalities
            modality_names = list(features.keys())
            for i, modality in enumerate(modality_names):
                if i not in available_indices:
                    features[modality] = torch.zeros_like(features[modality])

            # Forward pass
            logits = model(features, mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = (all_preds == all_labels).float().mean().item()

    from sklearn.metrics import f1_score

    f1_macro = f1_score(
        all_labels.numpy(), all_preds.numpy(), average="macro", zero_division=0
    )

    return {"accuracy": accuracy, "f1_macro": f1_macro}


def _compute_modality_importance(results, modality_names):
    """
    Compute relative importance of each modality.

    Importance = average performance with modality - average without modality
    """
    importance = {}

    for modality in modality_names:
        # Get performance with this modality
        with_scores = []
        without_scores = []

        for combo_name, metrics in results["all_combinations"].items():
            if modality in combo_name:
                with_scores.append(metrics["accuracy"])
            else:
                without_scores.append(metrics["accuracy"])

        if with_scores and without_scores:
            importance[modality] = np.mean(with_scores) - np.mean(
                without_scores
            )
        else:
            importance[modality] = 0.0

    # Normalize to [0, 1]
    total = sum(abs(v) for v in importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}

    return importance


def save_results_json(results, output_path):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multimodal fusion model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Directory to save results",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="analysis",
        help="Directory to save calibration plots",
    )
    parser.add_argument(
        "--missing_modality_test",
        action="store_true",
        help="Run missing modality robustness test",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on"
    )

    args = parser.parse_args()

    # Load model from checkpoint
    print(f"Loading model from: {args.checkpoint}")
    model = MultimodalFusionModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(args.device)

    # Get config from model
    config = model.config

    # Create dataloaders
    print("Creating dataloaders...")
    _, _, test_loader = create_dataloaders(
        dataset_name=config.dataset.name,
        data_dir=config.dataset.data_dir,
        modalities=config.dataset.modalities,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        chunk_size=_cfg_get(config.dataset, "chunk_size"),
        prefetch_shards=_cfg_get(config.dataset, "prefetch_shards", True),
        pin_memory=_cfg_get(config.dataset, "pin_memory"),
        persistent_workers=_cfg_get(config.dataset, "persistent_workers"),
        prefetch_factor=_cfg_get(config.dataset, "prefetch_factor"),
        chunk_cache_dir=_cfg_get(config.dataset, "chunk_cache_dir"),
    )

    # Standard evaluation
    print("\n" + "=" * 80)
    print("Standard Evaluation")
    print("=" * 80)

    metrics, (preds, labels, confidences, logits) = evaluate_model(
        model,
        test_loader,
        args.device,
        return_predictions=True,
        include_logits=True,
    )

    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"Test Loss: {metrics['loss']:.4f}")

    # Calibration metrics
    print("\nComputing calibration metrics...")
    ece = CalibrationMetrics.expected_calibration_error(
        confidences, preds, labels
    )
    mce = CalibrationMetrics.maximum_calibration_error(
        confidences, preds, labels
    )
    nll = CalibrationMetrics.negative_log_likelihood(logits, labels)
    print(f"ECE: {ece:.4f}")
    print(f"MCE: {mce:.4f}")
    print(f"NLL: {nll:.4f}")

    num_bins = int(
        _cfg_get(config.evaluation, "num_calibration_bins", 15)
    )
    analysis_root = Path(args.analysis_dir) / config.model.fusion_type
    analysis_root.mkdir(parents=True, exist_ok=True)
    calibration_plot = analysis_root / "calibration.png"
    CalibrationMetrics.reliability_diagram(
        confidences.cpu().numpy(),
        preds.cpu().numpy(),
        labels.cpu().numpy(),
        num_bins=num_bins,
        save_path=calibration_plot,
    )
    attention_plot = None
    if config.model.fusion_type == "hybrid":
        attention_plot = generate_attention_visualization(
            model,
            test_loader,
            list(config.dataset.modalities),
            analysis_root / "attention_viz.png",
            args.device,
        )
        if attention_plot is not None:
            print(f"Attention visualization saved to: {attention_plot}")

    # Inference latency
    print("\nMeasuring inference latency...")
    latency_mean_ms, latency_std_ms = measure_inference_latency(
        model, test_loader, args.device
    )
    print(
        f"Per-sample inference time: {latency_mean_ms:.3f} ± {latency_std_ms:.3f} ms"
    )

    # Save standard results
    standard_results = {
        "dataset": config.dataset.name,
        "fusion_type": config.model.fusion_type,
        "test_accuracy": metrics["accuracy"],
        "test_f1_macro": metrics["f1_macro"],
        "test_loss": metrics["loss"],
        "ece": ece,
        "mce": mce,
        "nll": nll,
        "inference_ms_mean": latency_mean_ms,
        "inference_ms_std": latency_std_ms,
    }
    if attention_plot is not None:
        standard_results["attention_plot"] = str(attention_plot)

    # Missing modality test
    if args.missing_modality_test:
        print("\n" + "=" * 80)
        print("Missing Modality Robustness Test")
        print("=" * 80)

        missing_results = evaluate_missing_modalities(
            model, test_loader, config.dataset.modalities, args.device
        )

        # Print summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(
            f"\nFull modalities: {missing_results['full_modalities']['accuracy']:.4f}"
        )
        print("\nSingle modality performance:")
        for modality, metrics in missing_results["single_modalities"].items():
            print(f"  {modality}: {metrics['accuracy']:.4f}")

        print("\nModality importance scores:")
        for modality, score in missing_results["modality_importance"].items():
            print(f"  {modality}: {score:.4f}")

        # Save missing modality results
        output_path = Path(args.output_dir) / "missing_modality.json"
        save_results_json(missing_results, output_path)

    # Save all results
    eval_path = Path(args.output_dir) / "evaluation_results.json"
    save_results_json(standard_results, eval_path)

    uncertainty_results = {
        "dataset": config.dataset.name,
        "fusion_type": config.model.fusion_type,
        "ece": ece,
        "mce": mce,
        "nll": nll,
        "num_bins": num_bins,
        "calibration_plot": str(calibration_plot),
    }
    uncertainty_path = Path(args.output_dir) / "uncertainty.json"
    save_results_json(uncertainty_results, uncertainty_path)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
