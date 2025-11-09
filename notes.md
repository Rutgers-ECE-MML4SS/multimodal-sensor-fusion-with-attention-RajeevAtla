# notes

## environment setup
- conda is really slow, probably significantly contributes to the 30 min time limit -> use uv instead

## data

- train-val-test split must be 0.7-0.15-0.15
- `data/preprocess.py` merges every PAMAP2 `.dat`, drops `activity_id=0`, and writes an aligned CSV after verifying each row has the 56-field schema.
- Heart-rate gaps are handled per subject via forward-fill/back-fill followed by a 25-sample rolling median to keep the signal smooth before training consumes it.
- Sharding: outputs live under `data/processed/subject_<id>/activity_<id>.csv` to keep files <20 MB for Git.
- Stratification: each run writes `data/splits/{train,val,test}.txt` where every line is `path/to/shard.csv,<rows>`, covering exactly 70/15/15 of total rows across subjects and activities.
- Tensor shards: preprocessing now also emits `data/processed_tensors/subject_<id>/activity_<id>.pt` so the loader can keep each shard in memory and avoid repeated CSV parsing.
- Loader: `src/data.py` now auto-detects the manifests, loads tensor shards (prefetching by default), slices modality-specific columns (e.g., `imu_hand → hand_*`, `heart_rate → heart_rate_bpm`), and falls back to the legacy `.npy` layout when manifests aren’t present.
- Chunking: dataset loader now uses `dataset.chunk_size` (default 1 024 timesteps) so each manifest shard is split into manageable sequence windows, keeping step counts low while still covering all samples.
- Training: gradient clipping is enforced via `training.gradient_clip_norm` (default 1.0) so Lightning clamps gradients during every optimizer step.
- Sequence batches: Each manifest chunk is treated as a single `[batch=1, seq_len, feature_dim]` sample so all IMU/HR streams flow through the `SequenceEncoder` path; labels stay constant per shard (activity ID).
- Numerical hygiene: manifest/Numpy payloads are sanitized with `torch.nan_to_num` inside `MultimodalDataset.__getitem__` so NaN/Inf rows from preprocessing don’t propagate into training loss.

## formatting/linting/type checking
- formatting and linting done with ruff
- type checking done with ty

## running (github actions)
- gh actions gives a pretty slim image - 16 gb ram, 2 vcpu cores (ubuntu-latest)
- can reduce further - 5 gb ram, 1 vcpu (ubuntu-slim); max execution time is 15 minutes however
- `complete_run.yml` now runs a quick fusion sweep (early/late/hybrid, 5 epochs each) via `uv run python src/train.py model.fusion_type=<type> training.max_epochs=5` to validate all heads without blowing the time budget.
- Lightning’s `configure_gradient_clipping` hook was updated to accept optional `optimizer_idx`/clip args, matching the latest API so CI runs don’t crash when clipping is enabled.
- `build_fusion_model` strips hybrid-only kwargs (e.g., `num_heads`) before instantiating Early/Late fusion, keeping a single Hydra config compatible with every architecture.
