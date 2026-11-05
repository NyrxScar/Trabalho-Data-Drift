"""Monitoring pipeline entry point: ``python -m src.main``.

Stages are decoupled as in a production MLOps setup:

1. **Data** - datasets are generated only if missing (or ``--regenerate-data``);
   in a real system this stage is the ingestion of a new production batch.
2. **Training** - runs only if no model artefact exists (or ``--retrain``).
   Otherwise the persisted ``models/model.joblib`` is reused as-is.
3. **Inference** - loads the saved model, verifies lineage and scores the
   reference and production data (no fitting).
4. **Monitoring** - drift tests, data quality, consolidation and the
   retraining recommendation, written to ``data/drift_summary.json``.
5. **Reporting** - HTML reports built by reading the audit log back.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src import config
from src import train_model
from src.artifacts import read_json
from src.decision import ModelMonitoring, PipelineReport, build_pipeline_report
from src.detect_drift import (
    FeatureDriftResult,
    detect_drift,
    detect_prediction_drift,
    load_dataset,
    validate_alpha,
)
from src.exceptions import DataValidationError, ModelArtifactError
from src.generate_data import datasets_exist, generate_datasets, save_datasets, validate_datasets
from src.inference import (
    InferenceResult,
    LoadedModel,
    load_model,
    model_exists,
    predict,
    save_predictions,
    verify_lineage,
)
from src.report_generator import generate_html_reports, write_json_summary

logger = logging.getLogger("src.main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Climate data drift monitoring pipeline.")
    parser.add_argument(
        "--seed", type=int, default=config.DEFAULT_SEED, help="seed used when generating/training"
    )
    parser.add_argument(
        "--alpha", type=float, default=config.DEFAULT_ALPHA, help="significance level"
    )
    parser.add_argument(
        "--regenerate-data", action="store_true", help="regenerate the datasets even if present"
    )
    parser.add_argument(
        "--retrain", action="store_true", help="train a new model even if an artefact exists"
    )
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    return parser.parse_args(argv)


def ensure_datasets(data_dir: Path, seed: int, regenerate: bool) -> bool:
    """Generate and save the datasets when needed. Returns True if generated."""
    if datasets_exist(data_dir) and not regenerate:
        logger.info("Reusing existing datasets in %s.", data_dir)
        return False
    logger.info("Generating datasets (seed=%d).", seed)
    reference, current = generate_datasets(seed)
    validate_datasets(reference, current)
    save_datasets(reference, current, data_dir, seed)
    return True


def ensure_model(reference_path: Path, models_dir: Path, seed: int, retrain: bool) -> bool:
    """Train and persist the model when needed. Returns True if trained."""
    if model_exists(models_dir) and not retrain:
        logger.info("Reusing persisted model artefact in %s (no training).", models_dir)
        return False
    logger.info("Training model (%s).", "forced by --retrain" if retrain else "no artefact found")
    train_model.run(reference_path, models_dir, seed)
    return True


def _data_seed(data_dir: Path) -> int | None:
    manifest = data_dir / config.DATASET_METADATA_FILE
    return read_json(manifest).get("seed") if manifest.is_file() else None


def build_model_monitoring(
    model: LoadedModel,
    reference_inference: InferenceResult,
    current_inference: InferenceResult,
    prediction_drift: FeatureDriftResult,
) -> ModelMonitoring:
    """Collect the model facts that feed the decision and the reports."""
    metadata = model.metadata
    return ModelMonitoring(
        model_version=metadata.model_version,
        trained_at=metadata.trained_at,
        algorithm=metadata.algorithm,
        training_data_sha256=metadata.training_data_sha256,
        lineage_verified=True,
        holdout_metrics=metadata.holdout_metrics,
        imputation_values=metadata.imputation_values,
        inference_rows=current_inference.rows,
        imputed_counts=current_inference.imputed_counts,
        reference_positive_rate=reference_inference.predicted_positive_rate,
        current_positive_rate=current_inference.predicted_positive_rate,
        reference_mean_score=reference_inference.mean_score,
        current_mean_score=current_inference.mean_score,
        prediction_drift=prediction_drift,
    )


def run_pipeline(
    seed: int = config.DEFAULT_SEED,
    alpha: float = config.DEFAULT_ALPHA,
    data_dir: Path = config.DATA_DIR,
    reports_dir: Path = config.REPORTS_DIR,
    models_dir: Path = config.MODELS_DIR,
    regenerate_data: bool = False,
    retrain: bool = False,
) -> PipelineReport:
    """Execute every stage in order and return the consolidated report."""
    validate_alpha(alpha)
    reference_path = data_dir / config.REFERENCE_DATA_FILE
    current_path = data_dir / config.CURRENT_DATA_FILE

    logger.info("[1/7] Data stage.")
    data_generated = ensure_datasets(data_dir, seed, regenerate_data)

    logger.info("[2/7] Loading and validating datasets.")
    reference = load_dataset(reference_path)
    current = load_dataset(current_path)
    validate_datasets(reference, current)

    logger.info("[3/7] Training stage.")
    model_trained = ensure_model(reference_path, models_dir, seed, retrain)

    logger.info("[4/7] Inference stage (loading persisted model).")
    model = load_model(models_dir)
    verify_lineage(model, reference_path)
    reference_inference = predict(model, reference)
    current_inference = predict(model, current)
    save_predictions(reference_inference, data_dir / config.REFERENCE_PREDICTIONS_FILE)
    save_predictions(current_inference, data_dir / config.CURRENT_PREDICTIONS_FILE)

    logger.info("[5/7] Statistical drift tests and data quality (alpha=%s).", alpha)
    results = detect_drift(reference, current, alpha)
    score = config.PREDICTION_SCORE_COLUMN
    prediction_drift = detect_prediction_drift(
        reference_inference.predictions[score], current_inference.predictions[score], alpha
    )
    monitoring = build_model_monitoring(
        model, reference_inference, current_inference, prediction_drift
    )
    report = build_pipeline_report(
        results,
        alpha,
        seed=_data_seed(data_dir),
        model=monitoring,
        data_generated_this_run=data_generated,
        model_trained_this_run=model_trained,
    )

    logger.info("[6/7] Writing JSON audit log.")
    summary_path = write_json_summary(report, data_dir / config.DRIFT_SUMMARY_FILE)

    logger.info("[7/7] Building HTML reports from the audit log.")
    generate_html_reports(summary_path, data_dir, reports_dir)

    logger.info(
        "Run %s finished: %d/%d features drifted; prediction drift=%s; action=%s.",
        report.run_id,
        report.summary.drifted_features,
        report.summary.total_features,
        prediction_drift.status,
        report.decision.action,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper returning a process exit code."""
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    args = parse_args(argv)
    try:
        run_pipeline(
            args.seed,
            args.alpha,
            args.data_dir,
            args.reports_dir,
            args.models_dir,
            args.regenerate_data,
            args.retrain,
        )
    except (DataValidationError, ModelArtifactError, FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
