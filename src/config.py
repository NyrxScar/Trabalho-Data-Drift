"""Centralized configuration for the data drift monitoring pipeline.

Every tunable parameter (paths, random seed, distribution parameters,
statistical significance, severity thresholds and retraining policy) lives
here so that the remaining modules contain no magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the project root, never machine-specific)
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
MODELS_DIR: Path = PROJECT_ROOT / "models"

REFERENCE_DATA_FILE: str = "reference_data.csv"
CURRENT_DATA_FILE: str = "current_data.csv"
DATASET_METADATA_FILE: str = "dataset_metadata.json"
REFERENCE_PREDICTIONS_FILE: str = "reference_predictions.csv"
CURRENT_PREDICTIONS_FILE: str = "current_predictions.csv"
DRIFT_SUMMARY_FILE: str = "drift_summary.json"
MODEL_FILE: str = "model.joblib"
MODEL_METADATA_FILE: str = "model_metadata.json"
EXEC_SUMMARY_FILE: str = "exec_summary.html"
DRIFT_REPORT_FILE: str = "drift_report.html"

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES: tuple[str, ...] = (
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation",
)
CATEGORICAL_FEATURE: str = "sky_condition"
SKY_CATEGORIES: tuple[str, ...] = ("clear", "cloudy", "rainy")
ALL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + (CATEGORICAL_FEATURE,)

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
DEFAULT_SEED: int = 42
REFERENCE_SAMPLE_SIZE: int = 5_000
CURRENT_SAMPLE_SIZE: int = 5_000

# Fraction of `temperature` values replaced by NaN in production only.
CURRENT_MISSING_FEATURE: str = "temperature"
CURRENT_MISSING_RATE: float = 0.05


@dataclass(frozen=True)
class NumericDistribution:
    """Parameters of the synthetic distributions used for one dataset.

    Attributes:
        temperature_mean: Mean of the normal temperature distribution (°C).
        temperature_std: Standard deviation of temperature (°C).
        humidity_mean: Mean of the normal relative humidity distribution (%).
        humidity_std: Standard deviation of humidity (%).
        wind_speed_shape: Weibull shape parameter for wind speed.
        wind_speed_scale: Weibull scale parameter for wind speed (km/h).
        precipitation_shape: Gamma shape parameter for precipitation.
        precipitation_scale: Gamma scale parameter for precipitation (mm).
    """

    temperature_mean: float
    temperature_std: float
    humidity_mean: float
    humidity_std: float
    wind_speed_shape: float
    wind_speed_scale: float
    precipitation_shape: float
    precipitation_scale: float


# Reference = expected behaviour during training (dry, warm season).
REFERENCE_DISTRIBUTION = NumericDistribution(
    temperature_mean=25.0,
    temperature_std=4.0,
    humidity_mean=60.0,
    humidity_std=10.0,
    wind_speed_shape=2.0,
    wind_speed_scale=12.0,
    precipitation_shape=0.8,
    precipitation_scale=2.5,
)

# Production = rainier, colder period. Wind speed is intentionally kept
# identical so that the pipeline also has a "no drift expected" control case.
CURRENT_DISTRIBUTION = NumericDistribution(
    temperature_mean=22.0,
    temperature_std=4.5,
    humidity_mean=75.0,
    humidity_std=10.0,
    wind_speed_shape=2.0,
    wind_speed_scale=12.0,
    precipitation_shape=1.5,
    precipitation_scale=4.0,
)

HUMIDITY_BOUNDS: tuple[float, float] = (0.0, 100.0)

REFERENCE_SKY_PROPORTIONS: Mapping[str, float] = MappingProxyType(
    {"clear": 0.60, "cloudy": 0.30, "rainy": 0.10}
)
CURRENT_SKY_PROPORTIONS: Mapping[str, float] = MappingProxyType(
    {"clear": 0.10, "cloudy": 0.30, "rainy": 0.60}
)

# Tolerance used when validating generated proportions / missing rates.
PROPORTION_TOLERANCE: float = 0.01

# --------------------------------------------------------------------------- #
# Target (ground truth) used to train the production model
# --------------------------------------------------------------------------- #
# Binary label "will it rain tomorrow?". It exists only in the reference
# (training) data: in production the ground truth arrives later, so the
# current batch is scored without labels, as in a real deployment.
TARGET_COLUMN: str = "rain_tomorrow"
# Labels come from an independent random stream so that adding them does not
# change the feature values drawn with the same seed.
TARGET_RNG_STREAM: int = 1


@dataclass(frozen=True)
class LabelModel:
    """Logistic ground-truth relationship between features and the label.

    ``logit = intercept + sum(coef * (value - center) / scale)`` plus the
    ``sky_condition`` offsets; ``P(rain_tomorrow = 1) = sigmoid(logit)``.
    """

    intercept: float = -1.0
    humidity_center: float = 60.0
    humidity_scale: float = 10.0
    humidity_coef: float = 0.9
    precipitation_center: float = 2.0
    precipitation_scale: float = 2.0
    precipitation_coef: float = 0.6
    temperature_center: float = 25.0
    temperature_scale: float = 4.0
    temperature_coef: float = -0.4
    rainy_coef: float = 1.5
    clear_coef: float = -0.8


LABEL_MODEL = LabelModel()

# --------------------------------------------------------------------------- #
# Model training / inference
# --------------------------------------------------------------------------- #
MODEL_ALGORITHM: str = "LogisticRegression"
MODEL_TEST_SIZE: float = 0.20
MODEL_MAX_ITER: int = 1_000
CLASSIFICATION_THRESHOLD: float = 0.5
PREDICTION_SCORE_COLUMN: str = "rain_probability"
PREDICTION_LABEL_COLUMN: str = "rain_predicted"

# --------------------------------------------------------------------------- #
# Statistical testing
# --------------------------------------------------------------------------- #
DEFAULT_ALPHA: float = 0.05
MIN_SAMPLES_FOR_TEST: int = 30

# Cochran's rule for the chi-square approximation.
MIN_EXPECTED_FREQUENCY: float = 5.0
MAX_SHARE_BELOW_MIN_EXPECTED: float = 0.20
MONTE_CARLO_PERMUTATIONS: int = 10_000


@dataclass(frozen=True)
class SeverityThresholds:
    """Effect-size thresholds that classify the severity of a detected drift.

    A drift is "moderate" when ``effect >= moderate`` and "high" when
    ``effect >= high``; below ``moderate`` it is "low".
    """

    moderate: float
    high: float


# KS statistic D (max distance between ECDFs, 0..1).
KS_SEVERITY = SeverityThresholds(moderate=0.10, high=0.20)
# Cramér's V for a 2 x k table (Cohen's conventions for df = 1).
CRAMERS_V_SEVERITY = SeverityThresholds(moderate=0.10, high=0.30)

# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
MISSING_WARNING_PCT: float = 1.0
MISSING_CRITICAL_PCT: float = 10.0


# --------------------------------------------------------------------------- #
# Retraining policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RetrainingPolicy:
    """Explicit, configurable rule used to recommend model retraining.

    Retraining is recommended when at least one of the drift criteria holds:

    * ``high_severity_drift_count`` or more features show high-severity drift;
    * the share of drifted features is ``>= drifted_share_threshold``;
    * the model output (prediction drift) shows high-severity drift and
      ``prediction_drift_triggers_retraining`` is enabled.

    Data-quality findings never trigger retraining by themselves; they raise
    an alert. If any feature reaches ``MISSING_CRITICAL_PCT`` the
    recommendation is to fix the data pipeline before retraining.
    """

    high_severity_drift_count: int = 1
    drifted_share_threshold: float = 0.50
    prediction_drift_triggers_retraining: bool = True
    missing_warning_pct: float = MISSING_WARNING_PCT
    missing_critical_pct: float = MISSING_CRITICAL_PCT


DEFAULT_RETRAINING_POLICY = RetrainingPolicy()

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
