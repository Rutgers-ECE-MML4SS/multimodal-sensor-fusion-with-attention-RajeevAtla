"""
Preprocess `dataset2.csv`: remove transient segments, create heart-rate violin plots,
align multi-sensor streams to heart-rate cadence, and impute missing data.

Usage:
    python data/preprocess.py

Outputs:
- `data/people_activity_heart_violin_grid.png`
- `data/preprocessed/dataset2_aligned.csv`
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_filtered_dataset(
    source_path: Path,
    id_column: str = "PeopleId",
) -> pd.DataFrame:
    """
    Load the dataset, remove transient rows, and validate expected columns.
    """
    df = pd.read_csv(source_path)

    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in {source_path}")

    if "activityID" in df.columns:
        df = df[df["activityID"] != "transient activities"]
        df = df[df["activityID"].notna()]

    return df.reset_index(drop=True)


def create_heart_rate_violin_grid(
    df: pd.DataFrame,
    id_column: str,
    activity_column: str,
    output_path: Path,
) -> None:
    """
    Create heart-rate violin plots grouped by person then activity.
    """
    feature_column = "heart_rate"
    if feature_column not in df.columns:
        raise ValueError(f"Column '{feature_column}' not found in dataframe.")

    unique_people = sorted(df[id_column].unique())
    unique_activities = sorted(df[activity_column].unique())

    n_people = len(unique_people)
    n_activities = len(unique_activities)

    fig, axes = plt.subplots(
        n_people,
        n_activities,
        figsize=(2.5 * n_activities, 2.5 * n_people),
        sharex=False,
        sharey=True,
    )

    if n_people == 1 and n_activities == 1:
        axes = [[axes]]
    elif n_people == 1:
        axes = [axes]
    elif n_activities == 1:
        axes = [[ax] for ax in axes]

    for i, people_id in enumerate(unique_people):
        for j, activity in enumerate(unique_activities):
            ax = axes[i][j]
            subset = df[
                (df[id_column] == people_id) & (df[activity_column] == activity)
            ]
            if subset.empty:
                ax.axis("off")
                continue

            values = subset[feature_column].dropna()
            if values.empty:
                ax.axis("off")
                continue

            violins = ax.violinplot(
                values,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )

            for body in violins["bodies"]:
                body.set_facecolor("#1f77b4")
                body.set_edgecolor("black")
                body.set_alpha(0.7)

            if "cmedians" in violins:
                violins["cmedians"].set_color("black")

            ax.set_title(f"P{people_id} | {activity}", fontsize=8)
            ax.set_xlabel("")
            ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(feature_column)
            else:
                ax.set_ylabel("")

    plt.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def add_time_metadata(
    df: pd.DataFrame,
    id_col: str = "PeopleId",
    sampling_hz: int = 100,
) -> pd.DataFrame:
    """
    Add sequential sample indices and time information per person.
    Assumes data for each person is already ordered chronologically.
    """
    augmented = df.copy()
    augmented["sample_idx"] = augmented.groupby(id_col).cumcount()

    time_ns = (
        (augmented["sample_idx"].astype(np.int64) * (1e9 / sampling_hz))
        .round()
        .astype("int64")
    )
    augmented["time_ns"] = time_ns
    augmented["time_delta"] = pd.to_timedelta(time_ns, unit="ns")
    augmented["time_seconds"] = augmented["time_delta"].dt.total_seconds()
    return augmented


def _numeric_feature_columns(
    df: pd.DataFrame,
    exclude: Tuple[str, ...],
) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def resample_to_target_frequency(
    df: pd.DataFrame,
    id_col: str = "PeopleId",
    activity_col: str = "activityID",
    target_hz: float = 9.0,
) -> pd.DataFrame:
    """
    Down-sample sensor streams to align with heart-rate cadence.
    """
    if "time_delta" not in df.columns:
        raise ValueError("time_delta column missing. Call add_time_metadata first.")

    target_period_ns = int(round(1e9 / target_hz))
    resample_freq = f"{target_period_ns}ns"

    numeric_cols = _numeric_feature_columns(
        df,
        exclude=(id_col, "sample_idx", "time_ns", "time_delta", "time_seconds"),
    )

    resampled_frames = []
    for people_id, person_df in df.groupby(id_col):
        person_df = person_df.sort_values("time_delta")
        person_df = person_df.set_index("time_delta")

        numeric_resampled = (
            person_df[numeric_cols]
            .resample(resample_freq)
            .mean()
        )

        activity_resampled = (
            person_df[[activity_col]]
            .resample(resample_freq)
            .ffill()
            .bfill()
        )

        combined = numeric_resampled.copy()
        combined[activity_col] = activity_resampled[activity_col]
        combined[id_col] = people_id
        combined["time_seconds"] = combined.index.total_seconds()

        resampled_frames.append(combined.reset_index())

    resampled_df = pd.concat(resampled_frames, ignore_index=True)
    return resampled_df


def hierarchical_heart_rate_imputation(
    df: pd.DataFrame,
    id_col: str = "PeopleId",
    activity_col: str = "activityID",
    target_col: str = "heart_rate",
) -> pd.DataFrame:
    """
    Impute missing heart-rate values using temporal and hierarchical statistics.
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found for imputation.")

    imputed = df.copy()

    # Step 1: linear interpolation within each person/activity sequence
    imputed[target_col] = (
        imputed.groupby([id_col, activity_col])[target_col]
        .apply(lambda s: s.interpolate(method="linear", limit_direction="both"))
        .reset_index(level=[0, 1], drop=True)
    )

    # Step 2: per person & activity median
    per_activity_median = (
        imputed.groupby([id_col, activity_col])[target_col]
        .transform("median")
    )
    imputed[target_col] = imputed[target_col].fillna(per_activity_median)

    # Step 3: per person median
    per_person_median = imputed.groupby(id_col)[target_col].transform("median")
    imputed[target_col] = imputed[target_col].fillna(per_person_median)

    # Step 4: per activity median
    per_activity_global = imputed.groupby(activity_col)[target_col].transform("median")
    imputed[target_col] = imputed[target_col].fillna(per_activity_global)

    # Step 5: global median fallback
    imputed[target_col] = imputed[target_col].fillna(imputed[target_col].median())
    return imputed


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    source_csv = base_dir / "dataset2.csv"
    figure_path = base_dir / "people_activity_heart_violin_grid.png"
    preprocessed_dir = base_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = preprocessed_dir / "dataset2_aligned.csv"

    filtered_df = load_filtered_dataset(source_csv)
    create_heart_rate_violin_grid(
        filtered_df,
        id_column="PeopleId",
        activity_column="activityID",
        output_path=figure_path,
    )

    augmented_df = add_time_metadata(filtered_df)
    resampled_df = resample_to_target_frequency(
        augmented_df,
        id_col="PeopleId",
        activity_col="activityID",
        target_hz=9.0,
    )
    imputed_df = hierarchical_heart_rate_imputation(
        resampled_df,
        id_col="PeopleId",
        activity_col="activityID",
        target_col="heart_rate",
    )

    imputed_df.to_csv(aligned_path, index=False)
