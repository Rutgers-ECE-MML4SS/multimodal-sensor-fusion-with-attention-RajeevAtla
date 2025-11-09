"""
Utility for merging the PAMAP2 raw .dat dumps into a single CSV with named columns.

Usage
-----
python data/preprocess.py \
    --raw-dir data/raw \
    --output data/processed.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence

import polars as pl
import torch

DATA_DIR = Path(__file__).parent
REPO_ROOT = DATA_DIR.parent
RAW_SUBDIR = DATA_DIR / "raw"
DEFAULT_OUTPUT = DATA_DIR / "processed"
TENSOR_OUTPUT = DATA_DIR / "processed_tensors"
SPLIT_DIR = DATA_DIR / "splits"
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_FILENAMES = {
    "train": Path("data/splits/train.txt"),
    "val": Path("data/splits/val.txt"),
    "test": Path("data/splits/test.txt"),
}
IMU_SENSORS: Sequence[str] = ("hand", "chest", "ankle")
AXES: Sequence[str] = ("x", "y", "z")
ORIENTATION_COMPONENTS: Sequence[str] = ("w", "x", "y", "z")
HR_ROLLING_WINDOW = 25  # ~2.5 s of samples at 10 Hz effective HR rate


def _sensor_columns(sensor: str) -> List[str]:
    """Return the ordered column names for a single IMU sensor block."""

    prefix = sensor.lower()
    cols = [
        f"{prefix}_temp_c",
        *[f"{prefix}_acc16_{axis}_ms2" for axis in AXES],
        *[f"{prefix}_acc6_{axis}_ms2" for axis in AXES],
        *[f"{prefix}_gyro_{axis}_rads" for axis in AXES],
        *[f"{prefix}_mag_{axis}_ut" for axis in AXES],
        *[f"{prefix}_orientation_{comp}" for comp in ORIENTATION_COMPONENTS],
    ]
    return cols


DATA_COLUMNS: List[str] = ["timestamp_s", "activity_id", "heart_rate_bpm"]
for sensor in IMU_SENSORS:
    DATA_COLUMNS.extend(_sensor_columns(sensor))

assert (
    len(DATA_COLUMNS) == 54
), f"Expected 54 raw columns, found {len(DATA_COLUMNS)} definitions."


def _discover_raw_files(raw_dir: Path) -> List[Path]:
    """Return the list of subject files that need to be merged."""

    return sorted(path for path in raw_dir.glob("*.dat") if path.is_file())


def _load_subject_file(path: Path) -> pl.DataFrame:
    """Read a single subject .dat file into a DataFrame with named columns."""

    df = pl.read_csv(
        path,
        separator=" ",
        has_header=False,
        new_columns=DATA_COLUMNS,
        null_values=["NaN"],
        truncate_ragged_lines=True,
        infer_schema_length=0,
    )
    subject_id = "".join(filter(str.isdigit, path.stem)) or path.stem
    float_casts = [
        pl.col(name).cast(pl.Float64, strict=False).alias(name)
        for name in DATA_COLUMNS
        if name != "activity_id"
    ]
    df = df.with_columns(float_casts)
    df = df.with_columns(pl.col("activity_id").cast(pl.Int64, strict=False))
    df = df.with_columns(
        [
            pl.lit(subject_id).alias("subject_id"),
        ]
    )
    return df.select(["subject_id", *DATA_COLUMNS])


def _interpolate_heart_rate(df: pl.DataFrame, window_size: int) -> pl.DataFrame:
    """Fill HR gaps per subject and smooth with a rolling median."""

    def _smooth(group: pl.DataFrame) -> pl.DataFrame:
        hr_series = group["heart_rate_bpm"].forward_fill().backward_fill()
        hr_smoothed = hr_series.rolling_median(
            window_size=window_size,
            min_samples=1,
        )
        return group.with_columns(pl.Series("heart_rate_bpm", hr_smoothed))

    return df.group_by("subject_id", maintain_order=True).map_groups(_smooth)


def _validate_csv(path: Path) -> None:
    """Ensure every row has the same number of columns as the header."""

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc
        expected = len(header)
        for idx, row in enumerate(reader, start=2):
            if len(row) != expected:
                raise ValueError(
                    f"Row {idx} in {path} has {len(row)} columns; expected {expected}"
                )


def _save_tensor_shard(group: pl.DataFrame, tensor_path: Path, stats: dict) -> None:
    """Persist shard as a torch tensor with column metadata."""

    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_df = group.select(DATA_COLUMNS)
    columns = numeric_df.columns
    values = numeric_df.to_numpy().astype("float32", copy=True)
    for idx, column in enumerate(columns):
        if column in {"timestamp_s", "activity_id"}:
            continue
        column_stats = stats.get(column)
        if not column_stats:
            continue
        mean = column_stats["mean"]
        std = column_stats["std"] or 1.0
        values[:, idx] = (values[:, idx] - mean) / std
    # ensure label and timestamp remain original (integers / seconds)
    if "activity_id" in columns:
        values[:, columns.index("activity_id")] = group["activity_id"].to_numpy()
    if "timestamp_s" in columns:
        values[:, columns.index("timestamp_s")] = group["timestamp_s"].to_numpy()
    payload = {
        "columns": columns,
        "data": torch.tensor(values, dtype=torch.float32),
    }
    torch.save(payload, tensor_path)


def _compute_normalization_stats(df: pl.DataFrame) -> dict:
    """Compute mean and std for each numeric column (excluding identifiers)."""

    stats = {}
    for column in DATA_COLUMNS:
        if column in {"timestamp_s", "activity_id"}:
            continue
        series = df[column]
        mean = float(series.mean())
        std = float(series.std(ddof=0) or 1.0)
        stats[column] = {"mean": mean, "std": std}
    return stats


def _materialize_shards(df: pl.DataFrame, csv_dir: Path, tensor_dir: Path, stats: dict) -> List[dict]:
    """Write CSV and tensor shards; return metadata."""

    csv_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    metadata: List[dict] = []
    for (subject, activity), group in df.group_by(["subject_id", "activity_id"]):
        csv_subject_dir = csv_dir / f"subject_{subject}"
        csv_subject_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_subject_dir / f"activity_{activity}.csv"
        group.write_csv(csv_path, line_terminator="\n")
        _validate_csv(csv_path)

        tensor_path = tensor_dir / f"subject_{subject}/activity_{activity}.pt"
        _save_tensor_shard(group, tensor_path, stats)

        metadata.append(
            {
                "subject": subject,
                "activity": activity,
                "rows": group.height,
                "csv_path": csv_path,
                "tensor_path": tensor_path,
            }
        )

    return metadata


def _stratified_split(shards: List[dict]) -> dict:
    """Split shard list into train/val/test manifests."""

    import random

    random.seed(42)
    splits = {"train": [], "val": [], "test": []}
    targets = {"train": TRAIN_FRACTION, "val": VAL_FRACTION, "test": TEST_FRACTION}
    shards_by_activity: dict[str, List[dict]] = {}
    for shard in shards:
        shards_by_activity.setdefault(str(shard["activity"]), []).append(shard)
    for activity, group in shards_by_activity.items():
        random.shuffle(group)
        total_rows = sum(s["rows"] for s in group)
        quotas = {k: targets[k] * total_rows for k in targets}
        for shard in group:
            split = max(quotas, key=quotas.get)
            splits[split].append(shard)
            quotas[split] -= shard["rows"]
    def totals() -> dict:
        return {k: sum(s["rows"] for s in splits[k]) for k in splits}
    grand_total = sum(s["rows"] for s in shards)
    desired = {
        "train": TRAIN_FRACTION * grand_total,
        "val": VAL_FRACTION * grand_total,
        "test": TEST_FRACTION * grand_total,
    }
    for _ in range(1000):
        current = totals()
        over_split = max(
            splits, key=lambda k: current[k] - desired[k]
        )
        under_split = min(
            splits, key=lambda k: current[k] - desired[k]
        )
        over_amount = current[over_split] - desired[over_split]
        under_amount = desired[under_split] - current[under_split]
        if over_amount <= 0 and under_amount <= 0:
            break
        donor_candidates = splits[over_split]
        if not donor_candidates:
            break
        donor = min(donor_candidates, key=lambda s: abs(s["rows"] - under_amount))
        splits[over_split].remove(donor)
        splits[under_split].append(donor)
    for split in splits:
        if not splits[split]:
            # move smallest shard from largest split
            largest = max(splits, key=lambda k: sum(s["rows"] for s in splits[k]))
            donor = min(splits[largest], key=lambda s: s["rows"])
            splits[largest].remove(donor)
            splits[split].append(donor)
    return splits


def _write_split_manifests(splits: dict) -> None:
    """Write train/val/test manifests listing shard paths."""

    for name, shards in splits.items():
        manifest_path = SPLIT_FILENAMES[name]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        entries = sorted(
            f"{s['tensor_path'].relative_to(REPO_ROOT).as_posix()},{s['rows']}"
            for s in shards
        )
        manifest_path.write_text("\n".join(entries))


def merge_raw_files(raw_dir: Path, output_path: Path) -> Path:
    """
    Merge every .dat file under ``raw_dir`` into ``output_path`` CSV shards.

    Parameters
    ----------
    raw_dir:
        Directory containing the raw PAMAP2 ``*.dat`` subject files.
    output_path:
        Destination directory containing CSV shards.

    Returns
    -------
    Path
        The path of the written CSV file.
    """

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    raw_files = _discover_raw_files(raw_dir)
    if not raw_files:
        raise FileNotFoundError(f"No .dat files found under {raw_dir}")

    frames = [_load_subject_file(path) for path in raw_files]
    combined = pl.concat(frames, how="vertical")
    combined = combined.filter(pl.col("activity_id") != 0)
    combined = combined.sort(["subject_id", "timestamp_s"])
    combined = _interpolate_heart_rate(combined, window_size=HR_ROLLING_WINDOW)
    stats = _compute_normalization_stats(combined)
    metadata = _materialize_shards(combined, output_path, TENSOR_OUTPUT, stats)
    splits = _stratified_split(metadata)
    _write_split_manifests(splits)
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Merge PAMAP2 raw sensor dumps into a labeled CSV file."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_SUBDIR,
        help="Directory with the raw *.dat files (default: data/raw).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for CSV shards (default: data/processed).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entrypoint used by ``python data/preprocess.py``."""

    args = parse_args()
    output = merge_raw_files(args.raw_dir, args.output)
    print(f"Wrote merged CSV to {output}")


if __name__ == "__main__":
    main()
