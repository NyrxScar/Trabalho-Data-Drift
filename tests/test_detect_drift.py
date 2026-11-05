"""Tests for src.detect_drift."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import detect_drift as dd
from src.exceptions import DataValidationError

RNG_SEED = 123


def _normal(name: str, mean: float, size: int = 1000, seed: int = RNG_SEED) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mean, 1.0, size), name=name)


# ---------------------------------------------------------------- numeric / KS
def test_ks_detects_shift_and_matches_scipy() -> None:
    reference, current = _normal("x", 0.0), _normal("x", 1.0, seed=RNG_SEED + 1)
    result = dd.detect_numeric_drift(reference, current, alpha=0.05)
    expected = stats.ks_2samp(reference, current)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.drift_detected is True
    assert result.status == dd.STATUS_DRIFT
    assert result.severity == dd.SEVERITY_HIGH
    assert result.effect_size == result.statistic


def test_ks_identical_samples_show_no_drift() -> None:
    reference = _normal("x", 0.0)
    result = dd.detect_numeric_drift(reference, reference.copy())
    assert result.statistic == 0.0
    assert result.p_value == pytest.approx(1.0)
    assert result.drift_detected is False
    assert result.severity == dd.SEVERITY_NONE
    assert "não comprova" in result.interpretation


def test_ks_ignores_nan_but_counts_them() -> None:
    reference = _normal("x", 0.0)
    current = _normal("x", 0.0, seed=RNG_SEED + 2)
    current.iloc[:50] = np.nan
    result = dd.detect_numeric_drift(reference, current)
    expected = stats.ks_2samp(reference, current.dropna())
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert np.isfinite(result.p_value)
    assert result.missing.current_count == 50
    assert result.missing.current_pct == pytest.approx(5.0)
    assert result.current_valid_count == 950
    assert result.current_count == 1000


def test_ks_not_tested_when_too_few_valid_samples() -> None:
    reference = _normal("x", 0.0)
    current = pd.Series([np.nan] * 100, name="x", dtype=float)
    result = dd.detect_numeric_drift(reference, current)
    assert result.status == dd.STATUS_NOT_TESTED
    assert result.p_value is None and result.statistic is None
    assert result.drift_detected is None
    assert result.error


def test_ks_rejects_non_numeric_series() -> None:
    with pytest.raises(DataValidationError, match="not numeric"):
        dd.detect_numeric_drift(pd.Series(["a"] * 40, name="x"), _normal("x", 0.0))


def test_invalid_statistical_output_is_not_reported_as_result(monkeypatch) -> None:
    fake = SimpleNamespace(statistic=np.nan, pvalue=np.nan)
    monkeypatch.setattr(dd.stats, "ks_2samp", lambda *_args, **_kw: fake)
    result = dd.detect_numeric_drift(_normal("x", 0.0), _normal("x", 1.0))
    assert result.status == dd.STATUS_NOT_TESTED
    assert result.p_value is None
    assert "não finitos" in result.error


@pytest.mark.parametrize(
    ("effect", "expected"),
    [(0.05, dd.SEVERITY_LOW), (0.15, dd.SEVERITY_MODERATE), (0.25, dd.SEVERITY_HIGH)],
)
def test_classify_severity(effect, expected) -> None:
    assert dd.classify_severity(effect, dd.config.KS_SEVERITY) == expected


# ---------------------------------------------------------- categorical / chi2
def _categorical(counts: dict[str, int]) -> pd.Series:
    values = [cat for cat, n in counts.items() for _ in range(n)]
    return pd.Series(values, name="sky_condition", dtype=object)


def test_chi2_matches_scipy_and_detects_drift() -> None:
    reference = _categorical({"clear": 600, "cloudy": 300, "rainy": 100})
    current = _categorical({"clear": 100, "cloudy": 300, "rainy": 600})
    result = dd.detect_categorical_drift(reference, current)
    expected = stats.chi2_contingency(
        [[600, 300, 100], [100, 300, 600]], correction=False
    )
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.details["degrees_of_freedom"] == 2
    assert result.details["observed_frequencies"]["current"] == {
        "clear": 100,
        "cloudy": 300,
        "rainy": 600,
    }
    assert result.details["cochran_conditions_met"] is True
    assert result.details["p_value_method"] == dd.CHI2_METHOD_ASYMPTOTIC
    assert result.drift_detected is True
    assert result.effect_size == pytest.approx(np.sqrt(expected.statistic / 2000))


def test_chi2_same_distribution_has_no_drift() -> None:
    reference = _categorical({"clear": 600, "cloudy": 300, "rainy": 100})
    result = dd.detect_categorical_drift(reference, reference.copy())
    assert result.statistic == pytest.approx(0.0)
    assert result.drift_detected is False


def test_contingency_table_keeps_zero_frequency_categories() -> None:
    reference = _categorical({"clear": 50, "cloudy": 50})
    current = _categorical({"cloudy": 100})
    table, unexpected = dd.build_contingency_table(reference, current, ("clear", "cloudy", "rainy"))
    assert list(table.columns) == ["clear", "cloudy", "rainy"]
    assert table.loc["current", "clear"] == 0
    assert table.loc["reference", "rainy"] == 0
    assert unexpected == []


def test_chi2_excludes_categories_absent_in_both_datasets() -> None:
    reference = _categorical({"clear": 50, "cloudy": 50})
    current = _categorical({"clear": 20, "cloudy": 80})
    result = dd.detect_categorical_drift(reference, current)
    assert result.details["excluded_zero_categories"] == ["rainy"]
    assert result.details["degrees_of_freedom"] == 1
    assert result.status in (dd.STATUS_DRIFT, dd.STATUS_NO_DRIFT)


def test_chi2_uses_monte_carlo_when_expected_frequencies_are_small() -> None:
    reference = _categorical({"clear": 30, "cloudy": 3, "rainy": 1})
    current = _categorical({"clear": 28, "cloudy": 4, "rainy": 2})
    result = dd.detect_categorical_drift(reference, current)
    assert result.details["cochran_conditions_met"] is False
    assert result.details["p_value_method"] == dd.CHI2_METHOD_MONTE_CARLO
    assert 0.0 < result.p_value <= 1.0
    assert result.drift_detected is False


def test_monte_carlo_p_value_is_small_for_strong_difference() -> None:
    observed = np.array([[20, 0, 0], [0, 10, 10]])
    statistic = stats.chi2_contingency(observed, correction=False).statistic
    p_value = dd.monte_carlo_chi2_pvalue(observed, statistic, n_permutations=2000)
    assert p_value < 0.01


def test_chi2_reports_unexpected_categories() -> None:
    reference = _categorical({"clear": 50, "cloudy": 50})
    current = _categorical({"clear": 40, "cloudy": 40, "snow": 20})
    result = dd.detect_categorical_drift(reference, current)
    assert result.details["unexpected_categories"] == ["snow"]
    assert "snow" in result.details["categories"]


def test_chi2_counts_missing_categories() -> None:
    reference = _categorical({"clear": 50, "cloudy": 50})
    current = pd.concat([_categorical({"clear": 45, "cloudy": 45}), pd.Series([None] * 10)])
    current.name = "sky_condition"
    result = dd.detect_categorical_drift(reference, current)
    assert result.missing.current_count == 10
    assert result.current_valid_count == 90


# ------------------------------------------------------------- orchestration
def test_detect_drift_on_generated_data(datasets) -> None:
    reference, current = datasets
    results = {r.feature: r for r in dd.detect_drift(reference, current)}
    assert set(results) == {"temperature", "humidity", "wind_speed", "precipitation", "sky_condition"}
    for feature in ("temperature", "humidity", "precipitation", "sky_condition"):
        assert results[feature].drift_detected is True
    # wind_speed has identical generating parameters in both datasets.
    assert results["wind_speed"].drift_detected is False
    assert results["temperature"].missing.current_count == 250
    assert results["sky_condition"].test_name == dd.CHI2_TEST_NAME
    assert all(r.test_name == dd.KS_TEST_NAME for r in results.values() if r.feature_type == "numeric")


def test_detect_drift_rejects_empty_dataset(datasets) -> None:
    reference, current = datasets
    with pytest.raises(DataValidationError, match="empty"):
        dd.detect_drift(reference, current.iloc[0:0])


def test_detect_drift_rejects_missing_column(datasets) -> None:
    reference, current = datasets
    with pytest.raises(DataValidationError, match="precipitation"):
        dd.detect_drift(reference, current.drop(columns="precipitation"))


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_detect_drift_rejects_invalid_alpha(datasets, alpha) -> None:
    with pytest.raises(ValueError, match="alpha"):
        dd.detect_drift(*datasets, alpha=alpha)


def test_load_dataset_errors(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        dd.load_dataset(tmp_path / "missing.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError):
        dd.load_dataset(empty)
    header_only = tmp_path / "header.csv"
    header_only.write_text("temperature,humidity\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="no rows"):
        dd.load_dataset(header_only)
