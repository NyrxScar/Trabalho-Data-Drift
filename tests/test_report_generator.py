"""Tests for src.report_generator and the end-to-end pipeline outputs."""

from __future__ import annotations

import copy
import json
import math
from html import escape

import numpy as np
import pandas as pd
import pytest

from src import config
from src import report_generator as rg
from src.artifacts import read_json, to_jsonable


def test_to_jsonable_normalises_special_values() -> None:
    payload = to_jsonable(
        {"nan": float("nan"), "inf": float("inf"), "np": np.float64(1.5), "t": (1, 2), 3: "x"}
    )
    assert payload == {"nan": None, "inf": None, "np": 1.5, "t": [1, 2], "3": "x"}
    json.dumps(payload, allow_nan=False)


def test_read_json_rejects_nan_literal(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"p": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="NaN"):
        read_json(path)


def test_format_p_value() -> None:
    assert rg.format_p_value(None) == rg.NOT_AVAILABLE
    assert rg.format_p_value(0.0) == "< 1e-300"
    assert rg.format_p_value(0.8223) == "0.8223"
    assert rg.format_p_value(3.7e-166) == "3.700e-166"
    assert rg.p_value_clause(0.0) == "p < 1e-300"
    assert rg.p_value_clause(0.5) == "p = 0.5000"


# ------------------------------------------------------------------- JSON log
def test_json_summary_matches_results(pipeline_run) -> None:
    report = pipeline_run.report
    data = read_json(pipeline_run.data_dir / config.DRIFT_SUMMARY_FILE)

    assert data["run"]["run_id"] == report.run_id
    assert data["run"]["timestamp"] == report.timestamp
    assert data["run"]["alpha"] == 0.05
    assert data["summary"]["total_features_analyzed"] == 5
    assert data["summary"]["features_with_drift"] == 4
    assert data["summary"]["drifted_features_pct"] == 80.0
    decision = data["retraining_decision"]
    assert decision["retraining_recommended"] is True
    assert decision["reasons"] == report.decision.reasons

    by_feature = {f["feature"]: f for f in data["features"]}
    for result in report.features:
        entry = by_feature[result.feature]
        assert entry["statistic"] == result.statistic
        assert entry["p_value"] == result.p_value
        assert entry["drift_detected"] == result.drift_detected
        assert entry["missing_values"]["current_count"] == result.missing.current_count
    temperature = by_feature["temperature"]
    assert temperature["has_missing_values"] is True
    assert temperature["missing_values"]["current_pct"] == 5.0
    assert temperature["data_quality_status"] == "warning"
    assert by_feature["humidity"]["has_missing_values"] is False
    assert by_feature["sky_condition"]["details"]["degrees_of_freedom"] == 2


def test_json_contains_model_section(pipeline_run) -> None:
    data = read_json(pipeline_run.data_dir / config.DRIFT_SUMMARY_FILE)
    model = data["model"]
    report_model = pipeline_run.report.model
    assert model["model_version"] == report_model.model_version
    assert model["lineage_verified"] is True
    assert model["imputed_counts"] == {"temperature": 250}
    assert model["prediction_drift"]["feature_type"] == "prediction"
    assert model["prediction_drift"]["p_value"] == report_model.prediction_drift.p_value
    assert data["run"]["model_trained_this_run"] is True


def test_json_has_no_invalid_numbers(pipeline_report) -> None:
    def walk(value) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, float):
            assert math.isfinite(value)

    walk(rg.build_summary_payload(pipeline_report))


def test_report_round_trips_through_json(pipeline_report) -> None:
    rebuilt = rg.report_from_payload(rg.build_summary_payload(pipeline_report))
    assert rebuilt == pipeline_report


def test_report_from_payload_rejects_unknown_schema(pipeline_report) -> None:
    payload = rg.build_summary_payload(pipeline_report)
    payload["schema_version"] = "0.1"
    with pytest.raises(ValueError, match="schema"):
        rg.report_from_payload(payload)
    del payload["summary"]
    payload["schema_version"] = rg.JSON_SCHEMA_VERSION
    with pytest.raises(ValueError, match="Malformed"):
        rg.report_from_payload(payload)


# ---------------------------------------------------------- executive summary
def test_exec_summary_displays_logged_values(pipeline_run) -> None:
    report = pipeline_run.report
    html = (pipeline_run.reports_dir / config.EXEC_SUMMARY_FILE).read_text(encoding="utf-8")
    assert report.run_id in html
    assert report.timestamp in html
    for result in report.features:
        assert result.feature in html
        assert rg.format_statistic(result.statistic) in html
        assert rg.format_p_value(result.p_value).replace("<", "&lt;") in html
    assert html.count("🚨 DRIFT") >= 4
    assert "✅ Estável" in html
    assert "250 registros (5.00%)" in html
    assert "retreinamento recomendado? SIM" in html
    assert report.model.model_version in html
    assert "artefato .joblib reutilizado" not in html  # first run trained the model
    for reason in report.decision.reasons:
        assert escape(reason) in html


def test_exec_summary_omits_missing_callout_when_no_missing(pipeline_report) -> None:
    clean = copy.deepcopy(pipeline_report)
    for quality in clean.data_quality:
        quality.current_missing_count = 0
    assert "Falha de Data Quality" not in rg.render_exec_summary(clean)


def test_exec_summary_without_model(pipeline_report) -> None:
    no_model = copy.deepcopy(pipeline_report)
    no_model.model = None
    assert "Nenhum modelo foi avaliado" in rg.render_exec_summary(no_model)


# ---------------------------------------------------------------- Plotly report
def test_plotly_figure_uses_persisted_data(pipeline_run) -> None:
    data_dir, report = pipeline_run.data_dir, pipeline_run.report
    reference = pd.read_csv(data_dir / config.REFERENCE_DATA_FILE)
    current = pd.read_csv(data_dir / config.CURRENT_DATA_FILE)
    ref_pred = pd.read_csv(data_dir / config.REFERENCE_PREDICTIONS_FILE)
    cur_pred = pd.read_csv(data_dir / config.CURRENT_PREDICTIONS_FILE)
    figure = rg.build_drift_figure(report, reference, current, ref_pred, cur_pred)

    histograms = [t for t in figure.data if t.type == "histogram"]
    assert len(histograms) == 10  # 4 numeric features + prediction scores, x2
    assert len(histograms[1].x) == current["temperature"].notna().sum() == 4750
    assert len(histograms[-1].x) == len(cur_pred)

    sky = next(r for r in report.features if r.feature == "sky_condition")
    bars = [t for t in figure.data if t.type == "bar"]
    expected = [sky.details["current_proportions"][c] for c in sky.details["categories"]]
    assert list(bars[1].y) == pytest.approx(expected)
    assert list(bars[-1].y) == pytest.approx([report.model.current_positive_rate])

    titles = [a.text for a in figure.layout.annotations]
    for result in [*report.features, report.model.prediction_drift]:
        assert any(rg.format_statistic(result.statistic) in title for title in titles)

    table = next(t for t in figure.data if t.type == "table")
    assert list(table.cells.values[2]) == [rg.format_p_value(r.p_value) for r in report.features]
    assert list(table.cells.values[3]).count("🚨 DRIFT") == 4


def test_plotly_figure_without_predictions_has_four_rows(pipeline_report, datasets) -> None:
    figure = rg.build_drift_figure(pipeline_report, *datasets)
    assert len([t for t in figure.data if t.type == "histogram"]) == 8


def test_drift_report_file_is_self_contained(pipeline_run) -> None:
    html = (pipeline_run.reports_dir / config.DRIFT_REPORT_FILE).read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert pipeline_run.report.run_id in html


# ------------------------------------------------------------- decoupling
def test_html_reports_can_be_rebuilt_from_json_only(pipeline_run, tmp_path) -> None:
    paths = rg.generate_html_reports(
        pipeline_run.data_dir / config.DRIFT_SUMMARY_FILE, pipeline_run.data_dir, tmp_path
    )
    original = (pipeline_run.reports_dir / config.EXEC_SUMMARY_FILE).read_text(encoding="utf-8")
    assert paths["exec_summary"].read_text(encoding="utf-8") == original
    assert paths["drift_report"].stat().st_size > 0


def test_full_pipeline_generates_every_artifact(pipeline_run) -> None:
    for path in (
        pipeline_run.data_dir / config.REFERENCE_DATA_FILE,
        pipeline_run.data_dir / config.CURRENT_DATA_FILE,
        pipeline_run.data_dir / config.DATASET_METADATA_FILE,
        pipeline_run.data_dir / config.REFERENCE_PREDICTIONS_FILE,
        pipeline_run.data_dir / config.CURRENT_PREDICTIONS_FILE,
        pipeline_run.data_dir / config.DRIFT_SUMMARY_FILE,
        pipeline_run.models_dir / config.MODEL_FILE,
        pipeline_run.models_dir / config.MODEL_METADATA_FILE,
        pipeline_run.reports_dir / config.EXEC_SUMMARY_FILE,
        pipeline_run.reports_dir / config.DRIFT_REPORT_FILE,
    ):
        assert path.is_file() and path.stat().st_size > 0, path
