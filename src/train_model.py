"""Model training stage (run once, or when retraining is approved).

Trains the "rain tomorrow" classifier on the labelled reference data and
persists two artefacts:

* ``models/model.joblib`` - the fitted scikit-learn pipeline (preprocessing
  and estimator together, so inference applies exactly the same transforms);
* ``models/model_metadata.json`` - version, training date, holdout metrics,
  imputation values and the SHA-256 of the training data (lineage).

Training is deliberately decoupled from inference: the monitoring pipeline
only loads the saved artefact (see :mod:`src.inference`) and never refits.

Usage::

    python -m src.train_model
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.artifacts import file_sha256, write_json
from src.detect_drift import load_dataset
from src.exceptions import DataValidationError

logger = logging.getLogger(__name__)

PREPROCESS_STEP = "preprocess"
ESTIMATOR_STEP = "classifier"
NUMERIC_TRANSFORMER = "numeric"


@dataclass(frozen=True)
class ModelMetadata:
    """Descriptive and lineage information stored next to the model file."""

    model_version: str
    trained_at: str
    algorithm: str
    sklearn_version: str
    features: list[str]
    target: str
    classification_threshold: float
    training_data_file: str
    training_data_sha256: str
    training_rows: int
    holdout_rows: int
    holdout_metrics: dict[str, float]
    imputation_values: dict[str, float]
    seed: int


def build_model_pipeline(seed: int = config.DEFAULT_SEED) -> Pipeline:
    """Create the untrained preprocessing + logistic regression pipeline.

    Numeric features: median imputation (production may contain NaN) and
    standardisation. Categorical feature: one-hot encoding over the known
    categories; unknown categories are encoded as all zeros instead of failing.
    """
    preprocess = ColumnTransformer(
        [
            (
                NUMERIC_TRANSFORMER,
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                list(config.NUMERIC_FEATURES),
            ),
            (
                "categorical",
                OneHotEncoder(categories=[list(config.SKY_CATEGORIES)], handle_unknown="ignore"),
                [config.CATEGORICAL_FEATURE],
            ),
        ]
    )
    estimator = LogisticRegression(max_iter=config.MODEL_MAX_ITER, random_state=seed)
    return Pipeline([(PREPROCESS_STEP, preprocess), (ESTIMATOR_STEP, estimator)])


def evaluate_model(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    """Compute classification metrics on a labelled holdout set."""
    scores = model.predict_proba(features)[:, 1]
    predicted = (scores >= config.CLASSIFICATION_THRESHOLD).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "observed_positive_rate": float(labels.mean()),
        "predicted_positive_rate": float(predicted.mean()),
    }


def _validate_training_data(reference: pd.DataFrame) -> None:
    required = [*config.ALL_FEATURES, config.TARGET_COLUMN]
    missing = [column for column in required if column not in reference.columns]
    if missing:
        raise DataValidationError(f"Training data lacks columns: {missing}.")
    if reference[config.TARGET_COLUMN].nunique() < 2:
        raise DataValidationError("Training labels must contain both classes.")


def train_model(
    reference_path: Path, seed: int = config.DEFAULT_SEED
) -> tuple[Pipeline, ModelMetadata]:
    """Fit the model on the reference data and evaluate it on a holdout split.

    Raises:
        DataValidationError: If the training data is invalid.
    """
    reference = load_dataset(reference_path)
    _validate_training_data(reference)
    features = reference[list(config.ALL_FEATURES)]
    labels = reference[config.TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=config.MODEL_TEST_SIZE, random_state=seed, stratify=labels
    )

    model = build_model_pipeline(seed).fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    imputer = model.named_steps[PREPROCESS_STEP].named_transformers_[NUMERIC_TRANSFORMER][
        "imputer"
    ]
    trained_at = datetime.now(timezone.utc)
    metadata = ModelMetadata(
        model_version=f"rain-model-{trained_at:%Y%m%dT%H%M%SZ}",
        trained_at=trained_at.isoformat(timespec="seconds"),
        algorithm=config.MODEL_ALGORITHM,
        sklearn_version=sklearn.__version__,
        features=list(config.ALL_FEATURES),
        target=config.TARGET_COLUMN,
        classification_threshold=config.CLASSIFICATION_THRESHOLD,
        training_data_file=reference_path.name,
        training_data_sha256=file_sha256(reference_path),
        training_rows=len(x_train),
        holdout_rows=len(x_test),
        holdout_metrics=metrics,
        imputation_values={
            name: float(value)
            for name, value in zip(config.NUMERIC_FEATURES, imputer.statistics_)
        },
        seed=seed,
    )
    logger.info(
        "Trained %s on %d rows: holdout ROC AUC=%.4f, accuracy=%.4f.",
        metadata.model_version,
        metadata.training_rows,
        metrics["roc_auc"],
        metrics["accuracy"],
    )
    return model, metadata


def save_model(
    model: Pipeline, metadata: ModelMetadata, models_dir: Path = config.MODELS_DIR
) -> tuple[Path, Path]:
    """Persist the fitted pipeline (joblib) and its metadata (JSON)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / config.MODEL_FILE
    joblib.dump(model, model_path)
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise OSError(f"Failed to write model to {model_path}.")
    metadata_path = write_json(asdict(metadata), models_dir / config.MODEL_METADATA_FILE)
    logger.info("Saved model artefacts: %s, %s.", model_path.name, metadata_path.name)
    return model_path, metadata_path


def run(
    reference_path: Path = config.DATA_DIR / config.REFERENCE_DATA_FILE,
    models_dir: Path = config.MODELS_DIR,
    seed: int = config.DEFAULT_SEED,
) -> ModelMetadata:
    """Train, evaluate and persist the model."""
    model, metadata = train_model(reference_path, seed)
    save_model(model, metadata, models_dir)
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    run()
