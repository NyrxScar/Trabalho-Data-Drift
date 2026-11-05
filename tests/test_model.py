"""Tests for the decoupled training (src.train_model) and inference (src.inference)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import config, inference, train_model
from src.artifacts import file_sha256, read_json
from src.exceptions import DataValidationError, ModelArtifactError
from src.generate_data import run as generate_run
from src.main import run_pipeline


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Datasets and a trained model in an isolated folder."""
    root = tmp_path_factory.mktemp("model")
    reference_path, current_path = generate_run(seed=42, output_dir=root / "data")
    metadata = train_model.run(reference_path, root / "models", seed=42)
    return root, reference_path, current_path, metadata


def test_training_persists_model_and_metadata(trained) -> None:
    root, reference_path, _, metadata = trained
    models_dir = root / "models"
    assert (models_dir / config.MODEL_FILE).is_file()
    stored = read_json(models_dir / config.MODEL_METADATA_FILE)
    assert stored["model_version"] == metadata.model_version
    assert stored["training_data_sha256"] == file_sha256(reference_path)
    assert stored["features"] == list(config.ALL_FEATURES)
    assert stored["training_rows"] + stored["holdout_rows"] == config.REFERENCE_SAMPLE_SIZE


def test_holdout_metrics_are_valid_and_better_than_chance(trained) -> None:
    metrics = trained[3].holdout_metrics
    assert 0.5 < metrics["roc_auc"] <= 1.0
    for name in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= metrics[name] <= 1.0


def test_imputation_values_are_training_medians(trained) -> None:
    _, reference_path, _, metadata = trained
    reference = pd.read_csv(reference_path)
    for feature, value in metadata.imputation_values.items():
        # The imputer is fitted on the training split, so it approximates the full median.
        assert value == pytest.approx(reference[feature].median(), rel=0.05)


def test_inference_loads_artifact_without_fitting(trained, monkeypatch) -> None:
    root, _, current_path, metadata = trained

    def forbid_fit(*_args, **_kwargs):
        raise AssertionError("inference must not fit the model")

    monkeypatch.setattr(Pipeline, "fit", forbid_fit)
    model = inference.load_model(root / "models")
    result = inference.predict(model, pd.read_csv(current_path))
    assert model.metadata.model_version == metadata.model_version
    assert result.rows == config.CURRENT_SAMPLE_SIZE
    scores = result.predictions[config.PREDICTION_SCORE_COLUMN]
    assert scores.between(0, 1).all() and scores.notna().all()


def test_inference_reports_imputed_missing_values(trained) -> None:
    root, _, current_path, _ = trained
    model = inference.load_model(root / "models")
    result = inference.predict(model, pd.read_csv(current_path))
    assert result.imputed_counts == {"temperature": 250}


def test_loaded_model_reproduces_trained_predictions(trained) -> None:
    root, reference_path, _, _ = trained
    model = inference.load_model(root / "models")
    frame = pd.read_csv(reference_path)
    first = inference.predict(model, frame).predictions
    second = inference.predict(inference.load_model(root / "models"), frame).predictions
    pd.testing.assert_frame_equal(first, second)


def test_inference_rejects_empty_or_incomplete_batch(trained) -> None:
    model = inference.load_model(trained[0] / "models")
    with pytest.raises(DataValidationError, match="empty"):
        inference.predict(model, pd.DataFrame(columns=list(config.ALL_FEATURES)))
    with pytest.raises(DataValidationError, match="humidity"):
        inference.predict(model, pd.read_csv(trained[2]).drop(columns="humidity"))


def test_load_model_fails_when_artifact_missing(tmp_path) -> None:
    with pytest.raises(ModelArtifactError, match="train_model"):
        inference.load_model(tmp_path)


def test_lineage_check_detects_changed_reference(trained, tmp_path) -> None:
    root, reference_path, _, _ = trained
    model = inference.load_model(root / "models")
    inference.verify_lineage(model, reference_path)
    changed = tmp_path / "reference_data.csv"
    changed.write_bytes(reference_path.read_bytes() + b"\n")
    with pytest.raises(ModelArtifactError, match="different version"):
        inference.verify_lineage(model, changed)


def test_training_rejects_data_without_label(tmp_path, datasets) -> None:
    path = tmp_path / "reference_data.csv"
    datasets[0].drop(columns=config.TARGET_COLUMN).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match=config.TARGET_COLUMN):
        train_model.train_model(path)


def test_pipeline_reuses_model_on_second_run(tmp_path, monkeypatch) -> None:
    dirs = {name: tmp_path / name for name in ("data", "reports", "models")}
    first = run_pipeline(
        data_dir=dirs["data"], reports_dir=dirs["reports"], models_dir=dirs["models"]
    )
    model_file = dirs["models"] / config.MODEL_FILE
    mtime = os.stat(model_file).st_mtime_ns

    def forbid_training(*_args, **_kwargs):
        raise AssertionError("training must not run when an artefact exists")

    monkeypatch.setattr(train_model, "run", forbid_training)
    second = run_pipeline(
        data_dir=dirs["data"], reports_dir=dirs["reports"], models_dir=dirs["models"]
    )
    assert first.model_trained_this_run and first.data_generated_this_run
    assert not second.model_trained_this_run and not second.data_generated_this_run
    assert os.stat(model_file).st_mtime_ns == mtime
    assert second.model.model_version == first.model.model_version
    assert np.isclose(second.model.current_positive_rate, first.model.current_positive_rate)


def test_regenerated_data_with_other_seed_requires_retraining(tmp_path) -> None:
    dirs = {name: tmp_path / name for name in ("data", "reports", "models")}
    kwargs = {"data_dir": dirs["data"], "reports_dir": dirs["reports"], "models_dir": dirs["models"]}
    run_pipeline(seed=42, **kwargs)
    with pytest.raises(ModelArtifactError, match="--retrain"):
        run_pipeline(seed=7, regenerate_data=True, **kwargs)
    report = run_pipeline(seed=7, regenerate_data=True, retrain=True, **kwargs)
    assert report.model_trained_this_run and report.model.lineage_verified
