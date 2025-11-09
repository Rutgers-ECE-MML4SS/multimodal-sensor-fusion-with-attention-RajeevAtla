"""
Uncertainty Quantification for Multimodal Fusion

Implements methods for estimating and calibrating confidence scores:
1. MC Dropout for epistemic uncertainty
2. Calibration metrics (ECE, reliability diagrams)
3. Uncertainty-weighted fusion
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropoutUncertainty(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.

    Runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty via variance.
    """

    model: nn.Module
    num_samples: int

    def __init__(self, model: nn.Module, num_samples: int = 10):
        """
        Args:
            model: The model to estimate uncertainty for
            num_samples: Number of MC dropout samples
        """
        super().__init__()
        cast_self = cast(Any, self)
        cast_self.model = model
        cast_self.num_samples = num_samples

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.

        Returns:
            mean_logits: (batch_size, num_classes) - mean prediction
            uncertainty: (batch_size,) - prediction uncertainty (variance)
        """
        was_training = self.model.training
        self.model.train()

        logits_samples = []
        prob_samples = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                logits = self.model(*args, **kwargs)
                probs = F.softmax(logits, dim=1)
                logits_samples.append(logits.unsqueeze(0))
                prob_samples.append(probs.unsqueeze(0))

        logits_stack = torch.cat(logits_samples, dim=0)
        probs_stack = torch.cat(prob_samples, dim=0)

        mean_logits = logits_stack.mean(dim=0)
        variance = probs_stack.var(dim=0, unbiased=False).mean(dim=1)

        if not was_training:
            self.model.eval()

        return mean_logits, variance


class CalibrationMetrics:
    """
    Compute calibration metrics for confidence estimates.

    Key metrics:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Negative Log-Likelihood (NLL)
    """

    @staticmethod
    def expected_calibration_error(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE = Σ (|bin_accuracy - bin_confidence|) * (bin_size / total_size)

        Args:
            confidences: (N,) - predicted confidence scores [0, 1]
            predictions: (N,) - predicted class indices
            labels: (N,) - ground truth class indices
            num_bins: Number of bins for calibration

        Returns:
            ece: Expected Calibration Error (lower is better)
        """
        confidences = confidences.detach().cpu()
        predictions = predictions.detach().cpu()
        labels = labels.detach().cpu()

        bin_bounds = torch.linspace(0.0, 1.0, steps=num_bins + 1)
        ece = torch.zeros(1, dtype=torch.float32)
        total = confidences.shape[0]

        for lower, upper in zip(bin_bounds[:-1], bin_bounds[1:]):
            if upper == 1.0:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)

            bin_count = in_bin.sum().item()
            if bin_count == 0:
                continue

            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = (
                (predictions[in_bin] == labels[in_bin]).float().mean()
            )
            ece += (bin_count / total) * torch.abs(
                bin_accuracy - bin_confidence
            )

        return float(ece.item())

    @staticmethod
    def maximum_calibration_error(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15,
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).

        MCE = max_bin |bin_accuracy - bin_confidence|

        Returns:
            mce: Maximum calibration error across bins
        """
        confidences = confidences.detach().cpu()
        predictions = predictions.detach().cpu()
        labels = labels.detach().cpu()

        bin_bounds = torch.linspace(0.0, 1.0, steps=num_bins + 1)
        max_error = torch.zeros(1, dtype=torch.float32)

        for lower, upper in zip(bin_bounds[:-1], bin_bounds[1:]):
            if upper == 1.0:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)

            if in_bin.sum() == 0:
                continue

            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = (
                (predictions[in_bin] == labels[in_bin]).float().mean()
            )
            bin_error = torch.abs(bin_accuracy - bin_confidence)
            max_error = torch.max(max_error, bin_error)

        return float(max_error.item())

    @staticmethod
    def negative_log_likelihood(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """
        Compute average Negative Log-Likelihood (NLL).

        NLL = -log P(y_true | x)

        Args:
            logits: (N, num_classes) - predicted logits
            labels: (N,) - ground truth labels

        Returns:
            nll: Average negative log-likelihood
        """
        logits = logits.detach()
        labels = labels.detach().to(dtype=torch.long)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        return float(loss.item())

    @staticmethod
    def reliability_diagram(
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 15,
        save_path: Path | str | None = None,
    ) -> None:
        """
        Plot reliability diagram showing calibration.

        X-axis: Predicted confidence
        Y-axis: Actual accuracy
        Perfect calibration: y = x (diagonal line)

        Args:
            confidences: (N,) - confidence scores
            predictions: (N,) - predicted classes
            labels: (N,) - ground truth
            num_bins: Number of bins
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt

        confidences = np.asarray(confidences)
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        bin_lowers = bin_edges[:-1]
        bin_uppers = bin_edges[1:]
        bin_centers = (bin_lowers + bin_uppers) / 2

        accuracies = np.zeros(num_bins, dtype=np.float32)
        avg_confidences = np.zeros(num_bins, dtype=np.float32)
        bin_counts = np.zeros(num_bins, dtype=np.float32)

        for idx, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
            if idx == num_bins - 1:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)

            count = np.sum(in_bin)
            bin_counts[idx] = count
            if count > 0:
                avg_confidences[idx] = confidences[in_bin].mean()
                accuracies[idx] = (predictions[in_bin] == labels[in_bin]).mean()

        width = 1.0 / num_bins
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(
            bin_centers,
            accuracies,
            width=width,
            alpha=0.7,
            edgecolor="black",
            label="Accuracy",
        )
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Reliability Diagram")

        ece = CalibrationMetrics.expected_calibration_error(
            torch.from_numpy(confidences),
            torch.from_numpy(predictions),
            torch.from_numpy(labels),
            num_bins=num_bins,
        )
        ax.text(
            0.02,
            0.95,
            f"ECE: {ece:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )
        ax.legend(loc="lower right")
        plt.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


class UncertaintyWeightedFusion(nn.Module):
    """
    Fuse modalities weighted by inverse uncertainty.

    Intuition: More uncertain modalities receive lower weight.
    Weights are proportional to 1 / (uncertainty_i + epsilon)
    """

    epsilon: float

    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small constant to avoid division by zero
        """
        super().__init__()
        cast_self = cast(Any, self)
        cast_self.epsilon = epsilon

    def forward(
        self,
        modality_predictions: Dict[str, torch.Tensor],
        modality_uncertainties: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse modality predictions weighted by inverse uncertainty.

        Args:
            modality_predictions: Dict of {modality: logits}
                                Each tensor: (batch_size, num_classes)
            modality_uncertainties: Dict of {modality: uncertainty}
                                   Each tensor: (batch_size,)
            modality_mask: (batch_size, num_modalities) - availability mask

        Returns:
            fused_logits: (batch_size, num_classes) - weighted fusion
            fusion_weights: (batch_size, num_modalities) - used weights
        """
        modality_names = list(modality_predictions.keys())
        if not modality_names:
            raise ValueError("No modality predictions supplied for fusion.")

        num_modalities = len(modality_names)
        device = next(iter(modality_predictions.values())).device
        mask = modality_mask.to(device=device, dtype=torch.float32)

        logits_stack = []
        weight_list = []
        for modality in modality_names:
            if modality not in modality_uncertainties:
                raise KeyError(
                    f"Missing uncertainty for modality '{modality}'."
                )

            logits = modality_predictions[modality].to(device)
            uncertainty = modality_uncertainties[modality].to(device)

            logits_stack.append(logits.unsqueeze(1))
            weight_list.append(1.0 / (uncertainty.unsqueeze(-1) + self.epsilon))

        logits_tensor = torch.cat(logits_stack, dim=1)
        raw_weights = torch.cat(weight_list, dim=1)
        weighted = raw_weights * mask

        weight_sums = weighted.sum(dim=1, keepdim=True)
        fallback = torch.where(
            mask.sum(dim=1, keepdim=True) > 0,
            mask / (mask.sum(dim=1, keepdim=True) + 1e-8),
            torch.full_like(mask, 1.0 / num_modalities),
        )
        fusion_weights = torch.where(
            weight_sums > 0, weighted / (weight_sums + 1e-8), fallback
        )

        fused_logits = (logits_tensor * fusion_weights.unsqueeze(-1)).sum(dim=1)
        return fused_logits, fusion_weights


class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration via temperature scaling.

    Learns a single temperature parameter T that scales logits:
    P_calibrated = softmax(logits / T)

    Reference: Guo et al. "On Calibration of Modern Neural Networks", ICML 2017
    """

    temperature: nn.Parameter

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: (batch_size, num_classes) - model outputs

        Returns:
            scaled_logits: (batch_size, num_classes) - temperature-scaled logits
        """
        return logits / self.temperature

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> None:
        """
        Learn optimal temperature on validation set.

        Args:
            logits: (N, num_classes) - validation set logits
            labels: (N,) - validation set labels
            lr: Learning rate
            max_iter: Maximum optimization iterations
        """

        logits = logits.detach()
        labels = labels.detach().to(dtype=torch.long)

        if logits.device != self.temperature.device:
            temp_param = self.temperature
            if temp_param.device.type == "meta":
                new_value = torch.ones(
                    temp_param.shape,
                    device=logits.device,
                    dtype=temp_param.dtype,
                )
            else:
                new_value = temp_param.detach().to(device=logits.device)
            self.temperature = nn.Parameter(new_value)

        self.temperature.data = torch.ones_like(self.temperature.data)

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature.data = self.temperature.data.clamp(min=1e-3)


class EnsembleUncertainty:
    """
    Estimate uncertainty via ensemble of models.

    Train multiple models with different initializations/data splits.
    Uncertainty = variance across ensemble predictions.
    """

    models: List[nn.Module]
    num_models: int

    def __init__(self, models: Sequence[nn.Module]):
        """
        Args:
            models: List of trained models (same architecture)
        """
        self.models = list(models)
        self.num_models = len(self.models)

    def predict_with_uncertainty(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and uncertainty from ensemble.

        Args:
            inputs: Model inputs

        Returns:
            mean_predictions: (batch_size, num_classes) - average prediction
            uncertainty: (batch_size,) - prediction variance
        """
        if self.num_models == 0:
            raise ValueError("Ensemble must contain at least one model.")

        prob_predictions = []
        with torch.no_grad():
            for model in self.models:
                was_training = model.training
                model.eval()
                logits = model(inputs)
                probs = F.softmax(logits, dim=1)
                prob_predictions.append(probs.unsqueeze(0))
                if was_training:
                    model.train()

        probs_tensor = torch.cat(
            prob_predictions, dim=0
        )  # (ensemble, batch, num_classes)
        mean_predictions = probs_tensor.mean(dim=0)
        uncertainty = probs_tensor.var(dim=0, unbiased=False).mean(dim=1)
        return mean_predictions, uncertainty


def compute_calibration_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute all calibration metrics on a dataset.

    Args:
        model: Trained model
        dataloader: Test/validation dataloader
        device: Device to run on

    Returns:
        metrics: Dict with ECE, MCE, NLL, accuracy
    """
    model.eval()
    all_confidences = []
    all_predictions = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

            all_confidences.append(confidences.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())

    if not all_confidences:
        raise ValueError("Dataloader produced no batches to evaluate.")

    confidences = torch.cat(all_confidences)
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    logits_tensor = torch.cat(all_logits)

    ece = CalibrationMetrics.expected_calibration_error(
        confidences, predictions, labels
    )
    mce = CalibrationMetrics.maximum_calibration_error(
        confidences, predictions, labels
    )
    nll = CalibrationMetrics.negative_log_likelihood(logits_tensor, labels)
    accuracy = (predictions == labels).float().mean().item()

    return {
        "ece": ece,
        "mce": mce,
        "nll": nll,
        "accuracy": accuracy,
    }


def main(
    save_path: Path | str = "test_reliability.png",
    num_samples: int = 1000,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """
    CLI entry point to demonstrate uncertainty calibration utilities.

    Args:
        save_path: Location to write the reliability diagram.
        num_samples: Number of synthetic samples to generate.
        num_classes: Number of classes for synthetic logits.

    Returns:
        Dictionary summarizing generated metrics and side effects.
    """
    print("Testing calibration metrics...")

    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)

    results: Dict[str, Any] = {"save_path": str(Path(save_path))}

    try:
        ece = CalibrationMetrics.expected_calibration_error(
            confidences, predictions, labels
        )
        print(f"✓ ECE computed: {ece:.4f}")
        results["ece"] = ece
    except NotImplementedError:
        print("✗ ECE not implemented yet")
        results["ece"] = None

    try:
        CalibrationMetrics.reliability_diagram(
            confidences.numpy(),
            predictions.numpy(),
            labels.numpy(),
            save_path=save_path,
        )
        print("✓ Reliability diagram created")
        results["diagram_created"] = True
    except NotImplementedError:
        print("✗ Reliability diagram not implemented yet")
        results["diagram_created"] = False

    return results


if __name__ == "__main__":
    main()
