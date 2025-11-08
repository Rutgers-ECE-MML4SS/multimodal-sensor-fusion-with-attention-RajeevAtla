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

RAW_SUBDIR = Path(__file__).parent / "raw"
DEFAULT_OUTPUT = Path(__file__).parent / "processed.csv"
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


def merge_raw_files(raw_dir: Path, output_csv: Path) -> Path:
    """
    Merge every .dat file under ``raw_dir`` into ``output_csv``.

    Parameters
    ----------
    raw_dir:
        Directory containing the raw PAMAP2 ``*.dat`` subject files.
    output_csv:
        Destination CSV path.

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
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(output_csv, line_terminator="\n")
    _validate_csv(output_csv)
    return output_csv


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
        help="Output CSV path (default: data/processed.csv).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entrypoint used by ``python data/preprocess.py``."""

    args = parse_args()
    output = merge_raw_files(args.raw_dir, args.output)
    print(f"Wrote merged CSV to {output}")


if __name__ == "__main__":
    main()
