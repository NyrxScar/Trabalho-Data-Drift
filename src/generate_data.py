"""Synthetic climate data generator.

Produces two datasets with the same schema:

* ``reference_data.csv`` - behaviour expected during model training, with
  the ground-truth label ``rain_tomorrow`` used to train the model;
* ``current_data.csv`` - data received in production, with an intentional
  distribution shift and a data-quality anomaly (missing ``temperature``).
  It has no label: in production the ground truth only arrives later.

A ``dataset_metadata.json`` manifest records the seed and the SHA-256 of each
file so downstream stages (training, monitoring) can verify data lineage.

Generation is fully reproducible through a configurable random seed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src import config
from src.artifacts import file_sha256, write_json
from src.exceptions import DataValidationError

logger = logging.getLogger(__name__)

DECIMALS: int = 2


def allocate_category_counts(
    sample_size: int, proportions: Mapping[str, float]
) -> dict[str, int]:
    """Convert proportions into integer counts that sum to ``sample_size``.

    Uses the largest-remainder method, so the realised proportions match the
    specification as closely as the sample size allows.

    Args:
        sample_size: Total number of records.
        proportions: Mapping category -> expected proportion (must sum to 1).

    Returns:
        Mapping category -> number of records.

    Raises:
        ValueError: If the inputs are invalid.
    """
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}.")
    if any(p < 0 for p in proportions.values()):
        raise ValueError("Category proportions must be non-negative.")
    if not np.isclose(sum(proportions.values()), 1.0):
        raise ValueError(
            f"Category proportions must sum to 1, got {sum(proportions.values())}."
        )

    raw = {cat: p * sample_size for cat, p in proportions.items()}
    counts = {cat: int(np.floor(value)) for cat, value in raw.items()}
    remaining = sample_size - sum(counts.values())
    by_remainder = sorted(raw, key=lambda cat: raw[cat] - counts[cat], reverse=True)
    for cat in by_remainder[:remaining]:
        counts[cat] += 1
    return counts


def generate_categorical(
    rng: np.random.Generator, sample_size: int, proportions: Mapping[str, float]
) -> np.ndarray:
    """Generate a shuffled categorical column with the given proportions."""
    counts = allocate_category_counts(sample_size, proportions)
    values = np.repeat(list(counts.keys()), list(counts.values()))
    return rng.permutation(values)


def generate_numeric_features(
    rng: np.random.Generator,
    sample_size: int,
    distribution: config.NumericDistribution,
) -> dict[str, np.ndarray]:
    """Draw the numeric climate variables from their parametric distributions.

    * temperature: normal;
    * humidity: normal clipped to the physical range [0, 100];
    * wind_speed: Weibull (typical for wind data);
    * precipitation: gamma (non-negative, right-skewed).
    """
    low, high = config.HUMIDITY_BOUNDS
    return {
        "temperature": rng.normal(
            distribution.temperature_mean, distribution.temperature_std, sample_size
        ),
        "humidity": np.clip(
            rng.normal(distribution.humidity_mean, distribution.humidity_std, sample_size),
            low,
            high,
        ),
        "wind_speed": distribution.wind_speed_scale
        * rng.weibull(distribution.wind_speed_shape, sample_size),
        "precipitation": rng.gamma(
            distribution.precipitation_shape,
            distribution.precipitation_scale,
            sample_size,
        ),
    }


def generate_dataset(
    rng: np.random.Generator,
    sample_size: int,
    distribution: config.NumericDistribution,
    sky_proportions: Mapping[str, float],
) -> pd.DataFrame:
    """Build one complete dataset (numeric + categorical features)."""
    frame = pd.DataFrame(generate_numeric_features(rng, sample_size, distribution))
    frame = frame.round(DECIMALS)
    frame[config.CATEGORICAL_FEATURE] = generate_categorical(
        rng, sample_size, sky_proportions
    )
    return frame[list(config.ALL_FEATURES)]


def generate_labels(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    label_model: config.LabelModel = config.LABEL_MODEL,
) -> np.ndarray:
    """Draw the binary label ``rain_tomorrow`` from a logistic relationship.

    Returns:
        Integer array of 0/1 labels, one per row of ``frame``.
    """
    m = label_model
    sky = frame[config.CATEGORICAL_FEATURE]
    logit = (
        m.intercept
        + m.humidity_coef * (frame["humidity"] - m.humidity_center) / m.humidity_scale
        + m.precipitation_coef
        * (frame["precipitation"] - m.precipitation_center)
        / m.precipitation_scale
        + m.temperature_coef * (frame["temperature"] - m.temperature_center) / m.temperature_scale
        + m.rainy_coef * (sky == "rainy")
        + m.clear_coef * (sky == "clear")
    ).to_numpy(dtype=float)
    probability = 1.0 / (1.0 + np.exp(-logit))
    return (rng.random(len(frame)) < probability).astype(int)


def inject_missing_values(
    frame: pd.DataFrame, column: str, rate: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``rate`` of ``column`` set to NaN.

    Exactly ``round(rate * len(frame))`` distinct rows are affected, chosen
    uniformly at random. The input frame is never modified.

    Raises:
        DataValidationError: If the column does not exist.
        ValueError: If ``rate`` is outside [0, 1].
    """
    if column not in frame.columns:
        raise DataValidationError(f"Column '{column}' not found in dataset.")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"Missing rate must be within [0, 1], got {rate}.")

    corrupted = frame.copy()
    n_missing = int(round(rate * len(corrupted)))
    rows = rng.choice(len(corrupted), size=n_missing, replace=False)
    corrupted.loc[corrupted.index[rows], column] = np.nan
    logger.info("Injected %d NaN values (%.1f%%) into '%s'.", n_missing, rate * 100, column)
    return corrupted


def generate_datasets(
    seed: int = config.DEFAULT_SEED,
    reference_size: int = config.REFERENCE_SAMPLE_SIZE,
    current_size: int = config.CURRENT_SAMPLE_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the reference (labelled) and production (unlabelled) datasets.

    One ``numpy.random.Generator`` seeded with ``seed`` drives the features and
    an independent stream drives the labels, so the output is reproducible.

    Returns:
        Tuple ``(reference, current)``.
    """
    rng = np.random.default_rng(seed)
    reference = generate_dataset(
        rng,
        reference_size,
        config.REFERENCE_DISTRIBUTION,
        config.REFERENCE_SKY_PROPORTIONS,
    )
    current = generate_dataset(
        rng, current_size, config.CURRENT_DISTRIBUTION, config.CURRENT_SKY_PROPORTIONS
    )
    current = inject_missing_values(
        current, config.CURRENT_MISSING_FEATURE, config.CURRENT_MISSING_RATE, rng
    )
    label_rng = np.random.default_rng([seed, config.TARGET_RNG_STREAM])
    reference[config.TARGET_COLUMN] = generate_labels(reference, label_rng)
    return reference, current


def _validate_proportions(
    frame: pd.DataFrame, expected: Mapping[str, float], name: str
) -> None:
    observed = frame[config.CATEGORICAL_FEATURE].value_counts(normalize=True)
    for category, expected_share in expected.items():
        share = float(observed.get(category, 0.0))
        if abs(share - expected_share) > config.PROPORTION_TOLERANCE:
            raise DataValidationError(
                f"{name}: '{category}' share {share:.3f} differs from "
                f"expected {expected_share:.3f}."
            )


def _validate_labels(reference: pd.DataFrame) -> None:
    if config.TARGET_COLUMN not in reference.columns:
        raise DataValidationError(f"reference dataset lacks label '{config.TARGET_COLUMN}'.")
    labels = set(reference[config.TARGET_COLUMN].unique())
    if labels != {0, 1}:
        raise DataValidationError(
            f"label '{config.TARGET_COLUMN}' must contain both classes 0 and 1, got {labels}."
        )


def validate_datasets(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Check that the generated datasets honour the specification.

    Verifies schema, non-emptiness, the reference label, category proportions,
    the missing rate injected in production and the absence of missing values
    in reference.

    Raises:
        DataValidationError: On the first violated rule.
    """
    for name, frame in (("reference", reference), ("current", current)):
        if frame.empty:
            raise DataValidationError(f"{name} dataset is empty.")
        missing_columns = set(config.ALL_FEATURES) - set(frame.columns)
        if missing_columns:
            raise DataValidationError(f"{name} dataset lacks columns {sorted(missing_columns)}.")

    _validate_labels(reference)
    _validate_proportions(reference, config.REFERENCE_SKY_PROPORTIONS, "reference")
    _validate_proportions(current, config.CURRENT_SKY_PROPORTIONS, "current")

    if reference.isna().any().any():
        raise DataValidationError("reference dataset must not contain missing values.")

    missing_rate = current[config.CURRENT_MISSING_FEATURE].isna().mean()
    if abs(missing_rate - config.CURRENT_MISSING_RATE) > config.PROPORTION_TOLERANCE:
        raise DataValidationError(
            f"current '{config.CURRENT_MISSING_FEATURE}' missing rate "
            f"{missing_rate:.3%} differs from expected {config.CURRENT_MISSING_RATE:.0%}."
        )
    logger.info(
        "Datasets validated: reference=%d rows, current=%d rows, "
        "current '%s' missing rate=%.2f%%.",
        len(reference),
        len(current),
        config.CURRENT_MISSING_FEATURE,
        missing_rate * 100,
    )


def datasets_exist(data_dir: Path = config.DATA_DIR) -> bool:
    """Return True when both dataset files are already present."""
    return all(
        (data_dir / name).is_file()
        for name in (config.REFERENCE_DATA_FILE, config.CURRENT_DATA_FILE)
    )


def save_datasets(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path = config.DATA_DIR,
    seed: int | None = None,
) -> tuple[Path, Path]:
    """Write both datasets as CSV plus a lineage manifest.

    Returns:
        Paths ``(reference_path, current_path)``.

    Raises:
        OSError: If a file could not be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (output_dir / config.REFERENCE_DATA_FILE, output_dir / config.CURRENT_DATA_FILE)
    for frame, path in zip((reference, current), paths):
        frame.to_csv(path, index=False)
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Failed to write dataset to {path}.")
        logger.info("Saved %s (%d rows).", path.name, len(frame))
    write_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": seed,
            "files": {
                path.name: {"rows": len(frame), "sha256": file_sha256(path)}
                for frame, path in zip((reference, current), paths)
            },
        },
        output_dir / config.DATASET_METADATA_FILE,
    )
    return paths


def run(seed: int = config.DEFAULT_SEED, output_dir: Path = config.DATA_DIR) -> tuple[Path, Path]:
    """Generate, validate and persist both datasets."""
    reference, current = generate_datasets(seed)
    validate_datasets(reference, current)
    return save_datasets(reference, current, output_dir, seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    run()
