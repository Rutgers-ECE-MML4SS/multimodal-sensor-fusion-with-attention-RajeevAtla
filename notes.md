# notes

## environment setup
- conda is really slow -> use uv instead

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
- Loader perf knobs:
  - `dataset.prefetch_shards` toggles whether manifest shards stay pinned in RAM.
  - `dataset.pin_memory` controls DataLoader pinning (handy for GPU experiments).
  - Chunk metadata is cached under `dataset.chunk_cache_dir` so reruns don’t recompute shard windows; delete the `.pt` files if chunk sizes change.

## formatting/linting/type checking
- formatting and linting done with `ruff`
- type checking done with `ty`

## running on ci
- GH runners provide ~16 GB RAM & 2 vCPUs (ubuntu-latest). 
- the merged workflow runs 13 short training jobs (3 fusion, 3 heads, 3 chunk sizes, 4 single-modality baselines).
- All training/analysis lives in `.github/workflows/parallel_run.yml`. Jobs:
  - `fusion-sweep`: early/late/hybrid (15 epochs); writes `experiments/<fusion>/` JSONs, `analysis/fusion/<fusion>/` plots, and uploads `artifacts/fusion/<fusion>/`.
  - `heads-ablation`: hybrid model with `{1,4,8}` heads; uploads `artifacts/heads/<value>/`.
  - `chunks-ablation`: hybrid model with chunk sizes `{256,512,1024}`; uploads `artifacts/chunk/<size>/`.
  - `single-modality-sweep`: early fusion with each modality alone; uploads `artifacts/single/<mod>/`.
- Each job packages its outputs into unique subfolders before upload (`artifacts/<group>/<config>/runs|experiments|analysis/...`). This avoids file collisions when artifacts are merged.
- The final `merge` job downloads all artifacts, copies their contents back into `runs/`, `experiments/`, and `analysis/`, rebuilds `experiments/fusion_comparison.json`, and calls `src/analysis.py --fusion_file ...` to regenerate the global comparison plot. Only this job pushes to git (guarded with a concurrency lock so it runs alone).
- To keep the workflow healthy:
  - Always respect the packaging layout if you add new jobs (copy into a unique `artifacts/<new_group>/<config>` prefix).
  - `src/eval.py` now writes `uncertainty.json`, `attention_viz.png`, calibration plots, and missing-modality JSONs for every run; make sure you don’t remove those hooks.
  - `analysis.py` supports `--fusion_file` so you can regenerate the aggregate plot locally: `uv run python src/analysis.py --experiment_dir experiments --output_dir analysis --fusion_file experiments/fusion_comparison.json`.
  
## steps for running locally
- install uv ([docs](https://docs.astral.sh/uv/getting-started/installation/))
- create venv - `uv venv`
- activate it - `.venv\Scipts\activate`
- sync dependencies - `uv sync`
- train a model - `uv python run src/train.py` (set whatever flags you need, defaults are in config/base.yml)
- run analysis `uv run python src/analysis.py --experiment_dir experiments --output_dir analysis --fusion_file experiments/fusion_comparison.json`
