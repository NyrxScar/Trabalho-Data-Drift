"""Statistical data drift detection.

* Numeric features: two-sample Kolmogorov-Smirnov test (``scipy.stats.ks_2samp``)
  computed on valid (non-missing) values only.
* Categorical feature: chi-square test of independence
  (``scipy.stats.chi2_contingency``) on a 2 x k contingency table
  (reference vs. production). When Cochran's conditions are not met, the
  p-value is obtained through a Monte Carlo permutation test instead.

Missing values are excluded from the tests but are always counted and
reported as an independent data-quality metric.

Every result is returned as a :class:`FeatureDriftResult`, which carries all
the information later needed by the reports and the retraining decision, so
no test ever has to be executed twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.exceptions import DataValidationError

logger = logging.getLogger(__name__)

STATUS_DRIFT = "drift"
STATUS_NO_DRIFT = "no_drift"
STATUS_NOT_TESTED = "not_tested"

SEVERITY_NONE = "none"
SEVERITY_LOW = "low"
SEVERITY_MODERATE = "moderate"
SEVERITY_HIGH = "high"

KS_TEST_NAME = "Kolmogorov-Smirnov (two-sample)"
CHI2_TEST_NAME = "Chi-square test of independence"
CHI2_METHOD_ASYMPTOTIC = "asymptotic"
CHI2_METHOD_MONTE_CARLO = "monte_carlo_permutation"


@dataclass(frozen=True)
class MissingValueStats:
    """Missing-value counts and percentages for one feature in both datasets."""

    reference_count: int
    reference_pct: float
    current_count: int
    current_pct: float


@dataclass
class FeatureDriftResult:
    """Outcome of the drift analysis for a single feature.

    ``statistic``, ``p_value``, ``drift_detected`` and ``effect_size`` are
    ``None`` when the test could not be executed (``status == "not_tested"``);
    in that case ``error`` explains why.
    """

    feature: str
    feature_type: str
    test_name: str
    alpha: float
    reference_count: int
    current_count: int
    reference_valid_count: int
    current_valid_count: int
    missing: MissingValueStats
    statistic: float | None = None
    p_value: float | None = None
    drift_detected: bool | None = None
    status: str = STATUS_NOT_TESTED
    effect_size_name: str | None = None
    effect_size: float | None = None
    severity: str = SEVERITY_NONE
    interpretation: str = ""
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset, failing loudly when it is missing or empty.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DataValidationError: If the file has no rows.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Dataset {path.name} is empty.") from exc
    if frame.empty:
        raise DataValidationError(f"Dataset {path.name} has no rows.")
    logger.info("Loaded %s: %d rows, %d columns.", path.name, *frame.shape)
    return frame


def validate_alpha(alpha: float) -> None:
    """Ensure the significance level lies strictly between 0 and 1."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")


def validate_inputs(
    reference: pd.DataFrame, current: pd.DataFrame, features: Sequence[str]
) -> None:
    """Check that both datasets are non-empty and contain every feature.

    Raises:
        DataValidationError: Describing the first problem found.
    """
    for name, frame in (("reference", reference), ("current", current)):
        if frame.empty:
            raise DataValidationError(f"{name} dataset is empty.")
        missing = [feature for feature in features if feature not in frame.columns]
        if missing:
            raise DataValidationError(f"{name} dataset lacks required columns: {missing}.")


def compute_missing_stats(reference: pd.Series, current: pd.Series) -> MissingValueStats:
    """Count missing values (NaN/None) in both series."""
    ref_missing = int(reference.isna().sum())
    cur_missing = int(current.isna().sum())
    return MissingValueStats(
        reference_count=ref_missing,
        reference_pct=_percentage(ref_missing, len(reference)),
        current_count=cur_missing,
        current_pct=_percentage(cur_missing, len(current)),
    )


def _percentage(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def classify_severity(effect_size: float, thresholds: config.SeverityThresholds) -> str:
    """Map an effect size to a severity label using configured thresholds."""
    if effect_size >= thresholds.high:
        return SEVERITY_HIGH
    if effect_size >= thresholds.moderate:
        return SEVERITY_MODERATE
    return SEVERITY_LOW


def interpret(p_value: float, alpha: float) -> str:
    """Produce a statistically careful reading of a p-value (report text, pt-BR)."""
    if p_value < alpha:
        return (
            f"p-value ({p_value:.3g}) < alpha ({alpha}): a hipótese nula de distribuições "
            "iguais é rejeitada; há evidência estatística de drift."
        )
    return (
        f"p-value ({p_value:.3g}) >= alpha ({alpha}): não há evidência estatística "
        "suficiente para rejeitar a hipótese nula. Isso não comprova que as "
        "distribuições sejam idênticas."
    )


def _finalise(
    result: FeatureDriftResult,
    statistic: float,
    p_value: float,
    effect_size: float,
    thresholds: config.SeverityThresholds,
) -> FeatureDriftResult:
    """Fill decision fields after a test ran, rejecting non-finite outputs."""
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        return _mark_not_tested(
            result, f"o teste retornou valores não finitos (estatística={statistic}, p={p_value})."
        )
    result.statistic = float(statistic)
    result.p_value = float(p_value)
    result.effect_size = float(effect_size)
    result.drift_detected = bool(p_value < result.alpha)
    result.status = STATUS_DRIFT if result.drift_detected else STATUS_NO_DRIFT
    result.severity = (
        classify_severity(effect_size, thresholds) if result.drift_detected else SEVERITY_NONE
    )
    result.interpretation = interpret(p_value, result.alpha)
    return result


def _mark_not_tested(result: FeatureDriftResult, reason: str) -> FeatureDriftResult:
    logger.warning("Feature '%s' not tested: %s", result.feature, reason)
    result.status = STATUS_NOT_TESTED
    result.error = reason
    result.interpretation = f"Teste não executado: {reason}"
    return result


def _base_result(
    feature: str,
    feature_type: str,
    test_name: str,
    reference: pd.Series,
    current: pd.Series,
    alpha: float,
) -> FeatureDriftResult:
    missing = compute_missing_stats(reference, current)
    return FeatureDriftResult(
        feature=feature,
        feature_type=feature_type,
        test_name=test_name,
        alpha=alpha,
        reference_count=len(reference),
        current_count=len(current),
        reference_valid_count=len(reference) - missing.reference_count,
        current_valid_count=len(current) - missing.current_count,
        missing=missing,
        effect_size_name="ks_statistic" if feature_type == "numeric" else "cramers_v",
    )


# --------------------------------------------------------------------------- #
# Numeric features - Kolmogorov-Smirnov
# --------------------------------------------------------------------------- #
def detect_numeric_drift(
    reference: pd.Series,
    current: pd.Series,
    alpha: float = config.DEFAULT_ALPHA,
    min_samples: int = config.MIN_SAMPLES_FOR_TEST,
) -> FeatureDriftResult:
    """Run the two-sample KS test on the valid values of a numeric feature.

    Raises:
        DataValidationError: If a series is not numeric.
    """
    validate_alpha(alpha)
    feature = str(reference.name)
    for name, series in (("reference", reference), ("current", current)):
        if not pd.api.types.is_numeric_dtype(series):
            raise DataValidationError(f"{name} '{feature}' is not numeric (dtype={series.dtype}).")

    result = _base_result(feature, "numeric", KS_TEST_NAME, reference, current, alpha)
    ref_valid = reference.dropna().to_numpy(dtype=float)
    cur_valid = current.dropna().to_numpy(dtype=float)
    result.details = {
        "reference_mean": _safe_stat(np.mean, ref_valid),
        "current_mean": _safe_stat(np.mean, cur_valid),
        "reference_std": _safe_stat(np.std, ref_valid),
        "current_std": _safe_stat(np.std, cur_valid),
        "reference_median": _safe_stat(np.median, ref_valid),
        "current_median": _safe_stat(np.median, cur_valid),
    }

    if min(len(ref_valid), len(cur_valid)) < min_samples:
        return _mark_not_tested(
            result,
            f"amostras válidas insuficientes (referência={len(ref_valid)}, "
            f"produção={len(cur_valid)}, mínimo={min_samples}).",
        )

    ks = stats.ks_2samp(ref_valid, cur_valid)
    # For KS, the statistic D itself is the effect size (max ECDF distance).
    return _finalise(result, ks.statistic, ks.pvalue, ks.statistic, config.KS_SEVERITY)


def _safe_stat(func: Any, values: np.ndarray) -> float | None:
    return float(func(values)) if len(values) else None


# --------------------------------------------------------------------------- #
# Categorical feature - Chi-square
# --------------------------------------------------------------------------- #
def build_contingency_table(
    reference: pd.Series, current: pd.Series, known_categories: Sequence[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Build a 2 x k table of observed frequencies (rows: reference/current).

    Every known category is present, even with zero frequency. Categories not
    listed in ``known_categories`` are appended as extra columns and returned
    separately so they can be reported as a quality issue.
    """
    ref_values = reference.dropna().astype(str)
    cur_values = current.dropna().astype(str)
    unexpected = sorted((set(ref_values) | set(cur_values)) - set(known_categories))
    categories = list(known_categories) + unexpected
    table = pd.DataFrame(
        [
            ref_values.value_counts().reindex(categories, fill_value=0),
            cur_values.value_counts().reindex(categories, fill_value=0),
        ],
        index=["reference", "current"],
    ).astype(int)
    return table, unexpected


def cochran_conditions_met(expected: np.ndarray) -> bool:
    """Return True when the chi-square approximation is considered reliable.

    Cochran's rule: no expected frequency below 1 and at most 20% of the
    cells with expected frequency below 5.
    """
    below_min = np.mean(expected < config.MIN_EXPECTED_FREQUENCY)
    return bool(expected.min() >= 1.0 and below_min <= config.MAX_SHARE_BELOW_MIN_EXPECTED)


def monte_carlo_chi2_pvalue(
    observed: np.ndarray,
    statistic: float,
    n_permutations: int = config.MONTE_CARLO_PERMUTATIONS,
    seed: int = config.DEFAULT_SEED,
) -> float:
    """Permutation p-value for the chi-square statistic of a 2 x k table.

    Group labels are shuffled while category labels stay fixed, which keeps
    both margins constant (so the expected table is unchanged).
    """
    rng = np.random.default_rng(seed)
    n_categories = observed.shape[1]
    labels = np.repeat(np.arange(n_categories), observed.sum(axis=0))
    n_reference = int(observed[0].sum())
    expected = stats.contingency.expected_freq(observed)
    column_totals = observed.sum(axis=0)

    exceed = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(labels)
        ref_counts = np.bincount(shuffled[:n_reference], minlength=n_categories)
        simulated = np.vstack([ref_counts, column_totals - ref_counts])
        if np.sum((simulated - expected) ** 2 / expected) >= statistic - 1e-12:
            exceed += 1
    return (exceed + 1) / (n_permutations + 1)


def detect_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    known_categories: Sequence[str] = config.SKY_CATEGORIES,
    alpha: float = config.DEFAULT_ALPHA,
    min_samples: int = config.MIN_SAMPLES_FOR_TEST,
) -> FeatureDriftResult:
    """Run a chi-square test of independence on a categorical feature.

    Yates' continuity correction is disabled so the statistic is the plain
    Pearson chi-square, consistent with Cramér's V and the permutation test.
    """
    validate_alpha(alpha)
    feature = str(reference.name)
    result = _base_result(feature, "categorical", CHI2_TEST_NAME, reference, current, alpha)
    table, unexpected = build_contingency_table(reference, current, known_categories)
    result.details = {
        "categories": list(table.columns),
        "observed_frequencies": table.to_dict(orient="index"),
        "reference_proportions": _proportions(table.loc["reference"]),
        "current_proportions": _proportions(table.loc["current"]),
        "unexpected_categories": unexpected,
    }
    if unexpected:
        logger.warning("Feature '%s' has unexpected categories: %s", feature, unexpected)

    if min(result.reference_valid_count, result.current_valid_count) < min_samples:
        return _mark_not_tested(
            result,
            f"amostras válidas insuficientes (referência={result.reference_valid_count}, "
            f"produção={result.current_valid_count}, mínimo={min_samples}).",
        )

    # Categories absent from both datasets carry no information and would
    # produce zero expected frequencies, so they are excluded from the test.
    tested = table.loc[:, table.sum(axis=0) > 0]
    result.details["excluded_zero_categories"] = [
        c for c in table.columns if c not in tested.columns
    ]
    if tested.shape[1] < 2:
        return _mark_not_tested(
            result, "menos de duas categorias observadas; não há o que comparar."
        )

    observed = tested.to_numpy()
    chi2 = stats.chi2_contingency(observed, correction=False)
    reliable = cochran_conditions_met(chi2.expected_freq)
    p_value = chi2.pvalue
    method = CHI2_METHOD_ASYMPTOTIC
    if not reliable:
        logger.warning(
            "Feature '%s': Cochran's conditions not met; using Monte Carlo permutation test.",
            feature,
        )
        p_value = monte_carlo_chi2_pvalue(observed, chi2.statistic)
        method = CHI2_METHOD_MONTE_CARLO

    n_total = observed.sum()
    cramers_v = np.sqrt(chi2.statistic / (n_total * (min(observed.shape) - 1)))
    result.details.update(
        {
            "degrees_of_freedom": int(chi2.dof),
            "expected_frequencies": pd.DataFrame(
                chi2.expected_freq, index=tested.index, columns=tested.columns
            )
            .round(4)
            .to_dict(orient="index"),
            "cochran_conditions_met": reliable,
            "p_value_method": method,
        }
    )
    return _finalise(result, chi2.statistic, p_value, cramers_v, config.CRAMERS_V_SEVERITY)


def _proportions(counts: pd.Series) -> dict[str, float]:
    total = float(counts.sum())
    return {str(k): (float(v) / total if total else 0.0) for k, v in counts.items()}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    alpha: float = config.DEFAULT_ALPHA,
    numeric_features: Sequence[str] = config.NUMERIC_FEATURES,
    categorical_feature: str = config.CATEGORICAL_FEATURE,
    known_categories: Sequence[str] = config.SKY_CATEGORIES,
) -> list[FeatureDriftResult]:
    """Run the appropriate drift test for every monitored feature.

    Raises:
        DataValidationError: If a dataset is empty or lacks a feature.
        ValueError: If ``alpha`` is invalid.
    """
    validate_alpha(alpha)
    validate_inputs(reference, current, [*numeric_features, categorical_feature])

    results = [
        detect_numeric_drift(reference[feature], current[feature], alpha)
        for feature in numeric_features
    ]
    results.append(
        detect_categorical_drift(
            reference[categorical_feature], current[categorical_feature], known_categories, alpha
        )
    )
    for result in results:
        logger.info(
            "%-14s | %-11s | stat=%s | p=%s | severity=%s | missing(current)=%.2f%%",
            result.feature,
            result.status,
            "n/a" if result.statistic is None else f"{result.statistic:.4f}",
            "n/a" if result.p_value is None else f"{result.p_value:.3e}",
            result.severity,
            result.missing.current_pct,
        )
    return results


def detect_prediction_drift(
    reference_scores: pd.Series,
    current_scores: pd.Series,
    alpha: float = config.DEFAULT_ALPHA,
) -> FeatureDriftResult:
    """Run the KS test on the model output scores (prediction drift).

    Comparing the score distribution on reference vs. production shows how
    much the input drift actually changes the model's behaviour, which is the
    signal available in production before ground-truth labels arrive.
    """
    result = detect_numeric_drift(reference_scores, current_scores, alpha)
    result.feature_type = "prediction"
    return result
