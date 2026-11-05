"""Tests for src.generate_data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.artifacts import file_sha256, read_json
from src.exceptions import DataValidationError
from src.generate_data import (
    allocate_category_counts,
    generate_datasets,
    inject_missing_values,
    run,
    validate_datasets,
)


def test_allocate_category_counts_matches_proportions_exactly() -> None:
    counts = allocate_category_counts(5000, config.REFERENCE_SKY_PROPORTIONS)
    assert counts == {"clear": 3000, "cloudy": 1500, "rainy": 500}


def test_allocate_category_counts_sums_to_size_with_rounding() -> None:
    counts = allocate_category_counts(7, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    assert sum(counts.values()) == 7


@pytest.mark.parametrize("proportions", [{"a": 0.5, "b": 0.6}, {"a": -0.1, "b": 1.1}])
def test_allocate_category_counts_rejects_invalid_proportions(proportions) -> None:
    with pytest.raises(ValueError):
        allocate_category_counts(100, proportions)


def test_datasets_have_expected_columns(datasets) -> None:
    reference, current = datasets
    assert list(reference.columns) == [*config.ALL_FEATURES, config.TARGET_COLUMN]
    # Production has no ground truth at inference time.
    assert list(current.columns) == list(config.ALL_FEATURES)
    assert len(datasets[0]) == config.REFERENCE_SAMPLE_SIZE
    assert len(datasets[1]) == config.CURRENT_SAMPLE_SIZE


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, config.REFERENCE_SKY_PROPORTIONS), (1, config.CURRENT_SKY_PROPORTIONS)],
)
def test_sky_condition_proportions(datasets, index, expected) -> None:
    observed = datasets[index][config.CATEGORICAL_FEATURE].value_counts(normalize=True)
    assert set(observed.index) == set(config.SKY_CATEGORIES)
    for category, share in expected.items():
        assert observed[category] == pytest.approx(share, abs=1e-9)


def test_missing_values_only_in_current_temperature(datasets) -> None:
    reference, current = datasets
    expected_missing = round(config.CURRENT_MISSING_RATE * len(current))
    assert current["temperature"].isna().sum() == expected_missing
    assert current["temperature"].isna().mean() == pytest.approx(0.05)
    assert current.drop(columns="temperature").isna().sum().sum() == 0
    assert reference.isna().sum().sum() == 0


def test_reference_labels_are_binary_and_follow_features(datasets) -> None:
    reference = datasets[0]
    labels = reference[config.TARGET_COLUMN]
    assert set(labels.unique()) == {0, 1}
    rain_rate = reference.groupby(config.CATEGORICAL_FEATURE)[config.TARGET_COLUMN].mean()
    assert rain_rate["rainy"] > rain_rate["cloudy"] > rain_rate["clear"]
    humid = reference["humidity"] > reference["humidity"].median()
    assert labels[humid].mean() > labels[~humid].mean()


def test_labels_do_not_change_feature_values() -> None:
    reference, _ = generate_datasets(seed=42)
    rng = np.random.default_rng(42)
    expected_temperature = rng.normal(
        config.REFERENCE_DISTRIBUTION.temperature_mean,
        config.REFERENCE_DISTRIBUTION.temperature_std,
        config.REFERENCE_SAMPLE_SIZE,
    ).round(2)
    np.testing.assert_array_equal(reference["temperature"].to_numpy(), expected_temperature)


def test_validate_datasets_requires_both_label_classes(datasets) -> None:
    reference, current = datasets
    tampered = reference.copy()
    tampered[config.TARGET_COLUMN] = 0
    with pytest.raises(DataValidationError, match="both classes"):
        validate_datasets(tampered, current)


def test_numeric_values_respect_physical_bounds(datasets) -> None:
    for frame in datasets:
        assert frame["humidity"].between(*config.HUMIDITY_BOUNDS).all()
        assert (frame["wind_speed"] >= 0).all()
        assert (frame["precipitation"] >= 0).all()


def test_same_seed_is_reproducible() -> None:
    first = generate_datasets(seed=7)
    second = generate_datasets(seed=7)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_data() -> None:
    first, _ = generate_datasets(seed=1)
    second, _ = generate_datasets(seed=2)
    assert not first.equals(second)


def test_inject_missing_values_does_not_mutate_input() -> None:
    frame = pd.DataFrame({"x": np.arange(100, dtype=float)})
    result = inject_missing_values(frame, "x", 0.1, np.random.default_rng(0))
    assert frame["x"].isna().sum() == 0
    assert result["x"].isna().sum() == 10


def test_inject_missing_values_rejects_unknown_column() -> None:
    with pytest.raises(DataValidationError):
        inject_missing_values(pd.DataFrame({"x": [1.0]}), "y", 0.1, np.random.default_rng(0))


def test_validate_datasets_detects_wrong_proportions(datasets) -> None:
    reference, current = datasets
    tampered = current.copy()
    tampered[config.CATEGORICAL_FEATURE] = "clear"
    with pytest.raises(DataValidationError, match="share"):
        validate_datasets(reference, tampered)


def test_validate_datasets_detects_missing_column_and_empty(datasets) -> None:
    reference, current = datasets
    with pytest.raises(DataValidationError, match="lacks columns"):
        validate_datasets(reference, current.drop(columns="humidity"))
    with pytest.raises(DataValidationError, match="empty"):
        validate_datasets(reference.iloc[0:0], current)


def test_validate_datasets_detects_nan_in_reference(datasets) -> None:
    reference, current = datasets
    tampered = reference.copy()
    tampered.loc[0, "humidity"] = np.nan
    with pytest.raises(DataValidationError, match="reference"):
        validate_datasets(tampered, current)


def test_run_writes_csv_files_that_round_trip(tmp_path) -> None:
    reference_path, current_path = run(seed=42, output_dir=tmp_path)
    assert reference_path.is_file() and current_path.is_file()
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    assert list(current.columns) == list(config.ALL_FEATURES)
    assert current["temperature"].isna().sum() == 250
    assert reference["temperature"].isna().sum() == 0
    manifest = read_json(tmp_path / config.DATASET_METADATA_FILE)
    assert manifest["seed"] == 42
    assert manifest["files"][config.REFERENCE_DATA_FILE]["sha256"] == file_sha256(reference_path)
    assert manifest["files"][config.CURRENT_DATA_FILE]["rows"] == len(current)
