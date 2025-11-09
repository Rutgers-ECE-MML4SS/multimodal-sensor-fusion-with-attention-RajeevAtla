"""
Training Script for Multimodal Sensor Fusion

Uses PyTorch Lightning for training with Hydra configuration.
Most infrastructure is provided - students need to integrate their fusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json
from typing import Any, Dict, List, cast

from data import create_dataloaders
from fusion import build_fusion_model
from encoders import build_encoder


class MultimodalFusionModule(pl.LightningModule):
    """
    PyTorch Lightning module for multimodal fusion training.

    Handles training loop, validation, and logging.
    """

    config: DictConfig
    encoders: nn.ModuleDict
    fusion_model: nn.Module
    criterion: nn.Module
    train_metrics: List[Dict[str, float]]
    val_metrics: List[Dict[str, float]]
    use_layer_norm: bool

    def __init__(self, config: DictConfig):
        """
        Args:
            config: Hydra configuration object
        """
        super().__init__()
        self.save_hyperparameters()
        cast_self = cast(Any, self)
        cast_self.config = config

        # Build encoders for each modality
        self.encoders = nn.ModuleDict()
        modality_output_dims = {}
        cast_self.use_layer_norm = bool(
            config.model.get("layer_norm", False)
        )
        self.layer_norms = nn.ModuleDict()

        for modality in config.dataset.modalities:
            encoder_config = dict(config.model.encoders.get(modality, {}))
            input_dim = encoder_config.pop("input_dim", 64)
            output_dim = config.model.output_dim

            self.encoders[modality] = build_encoder(
                modality=modality,
                input_dim=input_dim,
                output_dim=output_dim,
                encoder_config=encoder_config,
            )
            modality_output_dims[modality] = output_dim
            if self.use_layer_norm:
                self.layer_norms[modality] = nn.LayerNorm(output_dim)

        # Build fusion model
        # TODO: Students need to ensure their fusion implementation works here
        self.fusion_model = build_fusion_model(
            fusion_type=config.model.fusion_type,
            modality_dims=modality_output_dims,
            num_classes=config.dataset.get("num_classes", 11),
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.get("num_heads", 4),
            dropout=config.model.dropout,
        )

        # Loss function
        smoothing = config.training.get("label_smoothing", 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

        # Metrics storage
        cast_self.train_metrics = []
        cast_self.val_metrics = []

    def forward(self, features, mask=None):
        """
        Forward pass through encoders and fusion model.

        Args:
            features: Dict of {modality_name: features}
            mask: Optional modality availability mask

        Returns:
            logits: Class predictions
        """
        # Encode each modality
        encoded_features = {}
        for modality, encoder in self.encoders.items():
            if modality in features:
                encoded = encoder(features[modality])
                if self.use_layer_norm and modality in self.layer_norms:
                    encoded = self.layer_norms[modality](encoded)
                encoded_features[modality] = encoded

        # Fusion
        # TODO: Students ensure their fusion model returns correct format
        # For late fusion, may return tuple (logits, per_modality_logits)
        output = self.fusion_model(encoded_features, mask)

        # Handle different fusion output formats
        if isinstance(output, tuple):
            logits = output[
                0
            ]  # Late fusion returns (fused_logits, per_modality_logits)
        else:
            logits = output

        return logits

    def _log_metric(self, name: str, value: torch.Tensor, **kwargs: Any) -> None:
        """Call `self.log` only when the trainer reference is registered."""

        has_trainer = getattr(self, "_trainer", None) is not None
        has_fabric = getattr(self, "_fabric", None) is not None
        if not (has_trainer or has_fabric):
            return
        self.log(name, value, **kwargs)

    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        features, labels, mask = batch

        # Forward pass
        logits = self(features, mask)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self._log_metric(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self._log_metric(
            "train/acc", acc, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        features, labels, mask = batch

        # Forward pass
        logits = self(features, mask)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Get confidence for calibration
        probs = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)

        # Log metrics
        self._log_metric("val/loss", loss, on_epoch=True, prog_bar=True)
        self._log_metric("val/acc", acc, on_epoch=True, prog_bar=True)

        return {
            "val_loss": loss,
            "val_acc": acc,
            "preds": preds,
            "labels": labels,
            "confidences": confidences,
        }

    def test_step(self, batch, batch_idx):
        """Test step for one batch."""
        features, labels, mask = batch

        # Forward pass
        logits = self(features, mask)

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        probs = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)

        self._log_metric("test/acc", acc, on_epoch=True)

        return {"preds": preds, "labels": labels, "confidences": confidences}

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        if self.config.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {self.config.training.optimizer}"
            )

        # Learning rate scheduler
        if self.config.training.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.learning_rate / 100,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        elif self.config.training.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        else:
            return optimizer

    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx=None,
        gradient_clip_val=None,
        gradient_clip_algorithm="norm",
    ):
        """Ensure gradients are clipped deterministically every step."""

        if gradient_clip_val is not None and gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    """
    Main training function.

    Args:
        config: Hydra configuration
    """
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    # Set random seed for reproducibility
    pl.seed_everything(config.seed)

    # Create output directories
    save_dir = Path(config.experiment.save_dir) / config.experiment.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=config.dataset.name,
        data_dir=config.dataset.data_dir,
        modalities=config.dataset.modalities,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        modality_dropout=config.training.augmentation.modality_dropout,
        chunk_size=config.dataset.get("chunk_size"),
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = MultimodalFusionModule(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="{epoch}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=config.experiment.save_top_k,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=config.training.early_stopping_patience,
        mode="min",
        verbose=True,
    )

    # Logger
    logger = TensorBoardLogger(save_dir=save_dir, name="logs")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=config.experiment.log_every_n_steps,
        gradient_clip_val=config.training.gradient_clip_norm,
        deterministic=True,
        enable_progress_bar=True,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    # Test on best model
    print("\nTesting best model...")
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")

    trainer.test(model, test_loader, ckpt_path=best_model_path)

    # Save final results
    results = {
        "best_model_path": str(best_model_path),
        "best_val_loss": float(checkpoint_callback.best_model_score),
        "config": OmegaConf.to_container(config, resolve=True),
    }

    results_file = save_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete! Results saved to: {results_file}")
    print(f"Best model: {best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
