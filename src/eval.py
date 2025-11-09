"""
Evaluation Script for Multimodal Sensor Fusion

Provides framework for:
- Standard evaluation on test set
- Missing modality robustness testing
- Generating results for experiments/ directory
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import itertools
from typing import Any

from omegaconf import DictConfig

from train import MultimodalFusionModule
from data import create_dataloaders
from uncertainty import CalibrationMetrics


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Retrieve config values from DictConfig, dict, or objects."""

    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def evaluate_model(model, dataloader, device="cpu", return_predictions=False):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        return_predictions: If True, return all predictions and labels

    Returns:
        metrics: Dict with accuracy, loss, etc.
        predictions: Optional tuple of (preds, labels, confidences)
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_confidences = []
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
        return metrics, (all_preds, all_labels, all_confidences)
    else:
        return metrics


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

    metrics, (preds, labels, confidences) = evaluate_model(
        model, test_loader, args.device, return_predictions=True
    )

    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"Test Loss: {metrics['loss']:.4f}")

    # Calibration metrics
    print("\nComputing calibration metrics...")
    ece = CalibrationMetrics.expected_calibration_error(
        confidences, preds, labels
    )
    print(f"ECE: {ece:.4f}")

    # Save standard results
    standard_results = {
        "dataset": config.dataset.name,
        "fusion_type": config.model.fusion_type,
        "test_accuracy": metrics["accuracy"],
        "test_f1_macro": metrics["f1_macro"],
        "test_loss": metrics["loss"],
        "ece": ece,
    }

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
    output_path = Path(args.output_dir) / "evaluation_results.json"
    save_results_json(standard_results, output_path)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
