"""
Training Script for Multimodal Sensor Fusion

Uses PyTorch Lightning for training with Hydra configuration.
Most infrastructure is provided - students need to integrate their fusion models.
"""

import copy
import contextlib
import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
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
from typing import Any, Dict, List, Optional, cast
from collections import OrderedDict

from data import create_dataloaders
from fusion import build_fusion_model
from encoders import build_encoder


_COMPILED_MODULE_CACHE: "OrderedDict[str, nn.Module]" = OrderedDict()
_RESULT_WRITER = ThreadPoolExecutor(max_workers=1)


def _configure_torch_threads(max_threads: Optional[int] = None) -> None:
    """Clamp PyTorch thread usage to the available CPU budget."""

    available = os.cpu_count() or 1
    target = max_threads if max_threads is not None else available
    target = max(1, min(target, available))
    interop = max(1, target // 2)

    try:
        torch.set_num_threads(target)
    except Exception:  # pragma: no cover - platform dependent
        pass

    set_interop = getattr(torch, "set_num_interop_threads", None)
    if set_interop is not None:
        try:
            set_interop(interop)
        except Exception:  # pragma: no cover - platform dependent
            pass


def _configure_matmul_precision(mode: Optional[str] = None) -> None:
    """Set PyTorch matmul precision for float32 ops."""

    setter = getattr(torch, "set_float32_matmul_precision", None)
    if setter is None or not mode:
        return
    try:
        setter(mode)
    except Exception:  # pragma: no cover - platform dependent
        pass


def _clone_module(module: nn.Module) -> nn.Module:
    """Return a detached copy of a module, preserving type and buffers."""

    return copy.deepcopy(module)


def _remember_compiled_module(
    key: str, module: nn.Module, limit: int
) -> None:
    """Store a compiled module template for later reuse."""

    if limit <= 0:
        return
    snapshot = _clone_module(module).cpu()
    _COMPILED_MODULE_CACHE[key] = snapshot
    while len(_COMPILED_MODULE_CACHE) > limit:
        _COMPILED_MODULE_CACHE.popitem(last=False)


def _load_compiled_from_cache(key: str) -> Optional[nn.Module]:
    """Retrieve a cached compiled module copy."""

    cached = _COMPILED_MODULE_CACHE.get(key)
    if cached is None:
        return None
    return _clone_module(cached)


def _compile_with_cache(
    module: nn.Module,
    cache_key: str,
    backend: str,
    mode: str,
    cache_limit: int,
) -> nn.Module:
    """Compile a module with torch.compile, reusing cached graphs when possible."""

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return module

    cached = _load_compiled_from_cache(cache_key)
    if cached is not None:
        cached.load_state_dict(module.state_dict())
        cached.to(next(module.parameters()).device)
        return cached

    compiled = compile_fn(module, backend=backend, mode=mode)
    _remember_compiled_module(cache_key, compiled, cache_limit)
    return compiled


def _parse_manifest_paths(manifest_path: Path) -> List[Path]:
    """Return absolute shard paths referenced by a manifest file."""

    shard_paths: List[Path] = []
    if not manifest_path.exists():
        return shard_paths
    project_root = (
        manifest_path.parents[2]
        if len(manifest_path.parents) >= 3
        else Path(".")
    )
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "," not in line:
                continue
            path_str, _ = line.split(",", 1)
            shard_path = Path(path_str)
            if not shard_path.is_absolute():
                shard_path = (project_root / shard_path).resolve()
            shard_paths.append(shard_path)
    return shard_paths


def _stage_dataset_if_needed(config: DictConfig) -> Path:
    """Optionally copy referenced tensor shards to tmpfs and return new data dir."""

    dataset_cfg = config.dataset
    source_dir = Path(dataset_cfg.data_dir).resolve()
    if not bool(dataset_cfg.get("use_tmpfs", False)):
        return source_dir
    if os.name != "posix":
        return source_dir
    tmpfs_root = Path(dataset_cfg.get("tmpfs_root", "/dev/shm/a2_dataset"))
    if not tmpfs_root.exists():
        try:
            tmpfs_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            return source_dir

    target_dir = (tmpfs_root / source_dir.name).resolve()
    if target_dir.exists():
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    splits_src = source_dir / "splits"
    splits_dst = target_dir / "splits"
    if splits_src.exists():
        shutil.copytree(splits_src, splits_dst, dirs_exist_ok=True)

    manifest_files = [
        splits_dst / "train.txt",
        splits_dst / "val.txt",
        splits_dst / "test.txt",
    ]

    shard_paths: List[Path] = []
    for manifest in manifest_files:
        original_manifest = (
            splits_src / manifest.name if splits_src.exists() else manifest
        )
        shard_paths.extend(_parse_manifest_paths(original_manifest))

    copied = 0
    for shard in shard_paths:
        if not shard.exists():
            continue
        try:
            rel_path = shard.relative_to(source_dir)
        except ValueError:
            rel_path = shard.name
        dest = target_dir / rel_path
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shard, dest)
        copied += 1

    if copied == 0:
        # Fallback to copying entire directory if pattern selection failed
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    return target_dir


def _write_results_async(path: Path, payload: Dict[str, Any]) -> None:
    """Persist results JSON using the background executor."""

    def _writer(target: Path, data: Dict[str, Any]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    future = _RESULT_WRITER.submit(_writer, path, payload)
    future.result()


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
        self.autocast_enabled = bool(
            config.training.get("cpu_autocast", False)
        )
        self._maybe_compile_modules()

    def _maybe_compile_modules(self) -> None:
        """Compile encoders and fusion modules when torch.compile is available."""

        if not bool(self.config.training.get("enable_compile", True)):
            return

        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            return

        backend = self.config.training.get("compile_backend", "inductor")
        mode = self.config.training.get("compile_mode", "max-autotune")
        cache_limit = int(self.config.training.get("compile_cache_size", 0))

        for name, encoder in list(self.encoders.items()):
            cache_key = f"encoder::{name}::{encoder.__class__.__name__}"
            try:
                self.encoders[name] = _compile_with_cache(
                    encoder, cache_key, backend, mode, cache_limit
                )
            except Exception as exc:  # pragma: no cover - fallback path
                warnings.warn(
                    f"torch.compile failed for encoder '{name}': {exc}",
                    RuntimeWarning,
                )

        try:
            self.fusion_model = _compile_with_cache(
                self.fusion_model,
                f"fusion::{self.fusion_model.__class__.__name__}",
                backend,
                mode,
                cache_limit,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            warnings.warn(
                f"torch.compile failed for fusion model: {exc}",
                RuntimeWarning,
            )

    def _autocast_context(self):
        """Return autocast context for CPU/GPU depending on configuration."""

        if not self.autocast_enabled:
            return contextlib.nullcontext()
        device_type = (
            "cuda" if self.device.type == "cuda" else "cpu"
        )  # type: ignore[attr-defined]
        dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
        try:
            return torch.autocast(
                device_type=device_type, dtype=dtype, enabled=True
            )
        except RuntimeError:
            self.autocast_enabled = False
            return contextlib.nullcontext()

    def forward(self, features, mask=None):
        """
        Forward pass through encoders and fusion model.

        Args:
            features: Dict of {modality_name: features}
            mask: Optional modality availability mask

        Returns:
            logits: Class predictions
        """
        with self._autocast_context():
            # Encode each modality
            encoded_features = {}
            modality_order = list(self.encoders.keys())
            fold_size_cfg = int(self.config.model.get("modality_fold_size", 0))
            fold_size = (
                len(modality_order)
                if fold_size_cfg <= 0
                else max(1, min(fold_size_cfg, len(modality_order)))
            )

            for start in range(0, len(modality_order), fold_size):
                fold_modalities = modality_order[start : start + fold_size]
                for modality in fold_modalities:
                    if modality not in features:
                        continue
                    encoded = self.encoders[modality](features[modality])
                    if (
                        self.use_layer_norm
                        and modality in self.layer_norms
                    ):
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

    cpu_budget = min(4, os.cpu_count() or 4)
    _configure_torch_threads(cpu_budget)
    _configure_matmul_precision(config.training.get("matmul_precision"))
    dataset_data_dir = _stage_dataset_if_needed(config)

    # Set random seed for reproducibility
    pl.seed_everything(config.seed)

    # Create output directories
    save_dir = Path(config.experiment.save_dir) / config.experiment.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=config.dataset.name,
        data_dir=str(dataset_data_dir),
        modalities=config.dataset.modalities,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        modality_dropout=config.training.augmentation.modality_dropout,
        chunk_size=config.dataset.get("chunk_size"),
        prefetch_shards=config.dataset.get("prefetch_shards", True),
        pin_memory=config.dataset.get("pin_memory"),
        chunk_cache_dir=config.dataset.get("chunk_cache_dir"),
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
        save_on_train_epoch_end=False,
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
        accumulate_grad_batches=config.training.get(
            "gradient_accumulation", 1
        ),
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
    _write_results_async(results_file, results)

    print(f"\nTraining complete! Results saved to: {results_file}")
    print(f"Best model: {best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
