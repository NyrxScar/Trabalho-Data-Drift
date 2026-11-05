"""Inference stage: load the persisted model and score a batch of data.

This module never fits a model. It loads ``models/model.joblib`` (trained by
:mod:`src.train_model`), checks that the artefact is consistent with the
expected schema and with the reference data it was trained on, and produces
predictions.

Security note: ``joblib.load`` unpickles Python objects, so only artefacts
produced by this project's own training stage should ever be loaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.artifacts import file_sha256, read_json
from src.exceptions import DataValidationError, ModelArtifactError
from src.train_model import ModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """A fitted pipeline together with its metadata."""

    pipeline: Pipeline
    metadata: ModelMetadata


@dataclass
class InferenceResult:
    """Predictions for one batch plus the facts needed for monitoring."""

    predictions: pd.DataFrame
    rows: int
    imputed_counts: dict[str, int]
    predicted_positive_rate: float
    mean_score: float


def model_exists(models_dir: Path = config.MODELS_DIR) -> bool:
    """Return True when both model artefacts are present."""
    return all(
        (models_dir / name).is_file() for name in (config.MODEL_FILE, config.MODEL_METADATA_FILE)
    )


def load_model(models_dir: Path = config.MODELS_DIR) -> LoadedModel:
    """Load the persisted pipeline and metadata, validating both.

    Raises:
        ModelArtifactError: If a file is missing or the artefact is invalid.
    """
    model_path = models_dir / config.MODEL_FILE
    metadata_path = models_dir / config.MODEL_METADATA_FILE
    if not model_exists(models_dir):
        raise ModelArtifactError(
            f"Model artefacts not found in {models_dir}. Run 'python -m src.train_model' first."
        )
    try:
        metadata = ModelMetadata(**read_json(metadata_path))
    except (TypeError, ValueError) as exc:
        raise ModelArtifactError(f"Invalid model metadata in {metadata_path}: {exc}") from exc

    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "predict_proba"):
        raise ModelArtifactError(f"{model_path.name} is not a probabilistic sklearn Pipeline.")
    if metadata.features != list(config.ALL_FEATURES):
        raise ModelArtifactError(
            f"Model expects features {metadata.features}, "
            f"but the pipeline is configured for {list(config.ALL_FEATURES)}."
        )
    logger.info("Loaded model %s (trained at %s).", metadata.model_version, metadata.trained_at)
    return LoadedModel(pipeline=pipeline, metadata=metadata)


def verify_lineage(model: LoadedModel, reference_path: Path) -> None:
    """Ensure the model was trained on the current reference data.

    Raises:
        ModelArtifactError: If the reference file changed after training.
    """
    current_hash = file_sha256(reference_path)
    if current_hash != model.metadata.training_data_sha256:
        raise ModelArtifactError(
            f"Model {model.metadata.model_version} was trained on a different version of "
            f"{reference_path.name}. Retrain with 'python -m src.main --retrain'."
        )
    logger.info("Lineage verified: model trained on current %s.", reference_path.name)


def predict(model: LoadedModel, frame: pd.DataFrame) -> InferenceResult:
    """Score a batch with the loaded model (no fitting happens here).

    Missing numeric values are imputed inside the pipeline with the training
    medians; how many were imputed is reported for monitoring.

    Raises:
        DataValidationError: If the batch is empty or lacks model features.
    """
    features = model.metadata.features
    if frame.empty:
        raise DataValidationError("Cannot run inference on an empty batch.")
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise DataValidationError(f"Inference batch lacks model features: {missing}.")

    batch = frame[features]
    imputed = {
        name: int(count)
        for name, count in batch[list(config.NUMERIC_FEATURES)].isna().sum().items()
        if count > 0
    }
    scores = model.pipeline.predict_proba(batch)[:, 1]
    labels = (scores >= model.metadata.classification_threshold).astype(int)
    predictions = pd.DataFrame(
        {config.PREDICTION_SCORE_COLUMN: scores, config.PREDICTION_LABEL_COLUMN: labels},
        index=frame.index,
    )
    for name, count in imputed.items():
        logger.warning(
            "Inference imputed %d missing '%s' values with training median %.2f.",
            count,
            name,
            model.metadata.imputation_values[name],
        )
    return InferenceResult(
        predictions=predictions,
        rows=len(predictions),
        imputed_counts=imputed,
        predicted_positive_rate=float(labels.mean()),
        mean_score=float(scores.mean()),
    )


def save_predictions(result: InferenceResult, path: Path) -> Path:
    """Persist predictions as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(path, index=False)
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Failed to write predictions to {path}.")
    logger.info("Saved %s (%d rows).", path.name, result.rows)
    return path
