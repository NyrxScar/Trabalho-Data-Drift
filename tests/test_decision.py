"""Tests for src.decision (data quality, consolidation and retraining rule)."""

from __future__ import annotations

from src import decision as dec
from src.config import RetrainingPolicy
from src.detect_drift import (
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_NONE,
    STATUS_DRIFT,
    STATUS_NO_DRIFT,
    FeatureDriftResult,
    MissingValueStats,
)


def _result(feature: str, status: str, severity: str, missing_pct: float = 0.0) -> FeatureDriftResult:
    missing_count = int(missing_pct * 10)
    return FeatureDriftResult(
        feature=feature,
        feature_type="numeric",
        test_name="ks",
        alpha=0.05,
        reference_count=1000,
        current_count=1000,
        reference_valid_count=1000,
        current_valid_count=1000 - missing_count,
        missing=MissingValueStats(0, 0.0, missing_count, missing_pct),
        statistic=0.1,
        p_value=0.01 if status == STATUS_DRIFT else 0.5,
        drift_detected=status == STATUS_DRIFT,
        status=status,
        severity=severity,
    )


def _decide(results: list[FeatureDriftResult]) -> dec.RetrainingDecision:
    report = dec.build_pipeline_report(results, alpha=0.05)
    return report.decision


def test_no_drift_no_issue_means_no_action() -> None:
    decision = _decide([_result(f"f{i}", STATUS_NO_DRIFT, SEVERITY_NONE) for i in range(5)])
    assert decision.action == dec.ACTION_NO_ACTION
    assert decision.alert_level == dec.ALERT_NONE
    assert decision.retraining_recommended is False


def test_single_low_severity_drift_only_monitors() -> None:
    results = [_result("f0", STATUS_DRIFT, SEVERITY_LOW)] + [
        _result(f"f{i}", STATUS_NO_DRIFT, SEVERITY_NONE) for i in range(1, 5)
    ]
    decision = _decide(results)
    assert decision.retraining_recommended is False
    assert decision.action == dec.ACTION_MONITOR
    assert decision.alert_level == dec.ALERT_WARNING


def test_high_severity_drift_triggers_retraining() -> None:
    results = [_result("f0", STATUS_DRIFT, SEVERITY_HIGH)] + [
        _result(f"f{i}", STATUS_NO_DRIFT, SEVERITY_NONE) for i in range(1, 5)
    ]
    decision = _decide(results)
    assert decision.retraining_recommended is True
    assert decision.action == dec.ACTION_RETRAIN
    assert any("severidade alta" in reason for reason in decision.reasons)


def test_drifted_share_triggers_retraining_even_with_low_severity() -> None:
    results = [_result(f"f{i}", STATUS_DRIFT, SEVERITY_LOW) for i in range(3)] + [
        _result(f"f{i}", STATUS_NO_DRIFT, SEVERITY_NONE) for i in range(3, 5)
    ]
    decision = _decide(results)
    assert decision.retraining_recommended is True
    assert any("60.0%" in reason for reason in decision.reasons)


def test_missing_values_alone_do_not_trigger_retraining() -> None:
    results = [_result("temperature", STATUS_NO_DRIFT, SEVERITY_NONE, missing_pct=5.0)]
    decision = _decide(results)
    assert decision.retraining_recommended is False
    assert decision.action == dec.ACTION_INVESTIGATE_DATA


def test_quality_thresholds() -> None:
    results = [
        _result("ok", STATUS_NO_DRIFT, SEVERITY_NONE, missing_pct=0.5),
        _result("warn", STATUS_NO_DRIFT, SEVERITY_NONE, missing_pct=5.0),
        _result("crit", STATUS_NO_DRIFT, SEVERITY_NONE, missing_pct=15.0),
    ]
    quality = {q.feature: q.status for q in dec.assess_data_quality(results)}
    assert quality == {"ok": "ok", "warn": "warning", "crit": "critical"}


def test_critical_quality_blocks_direct_retraining() -> None:
    results = [_result("f0", STATUS_DRIFT, SEVERITY_HIGH, missing_pct=20.0)]
    decision = _decide(results)
    assert decision.action == dec.ACTION_FIX_THEN_RETRAIN


def test_policy_is_configurable() -> None:
    results = [_result("f0", STATUS_DRIFT, SEVERITY_HIGH)] + [
        _result(f"f{i}", STATUS_NO_DRIFT, SEVERITY_NONE) for i in range(1, 5)
    ]
    strict = RetrainingPolicy(high_severity_drift_count=2, drifted_share_threshold=0.9)
    report = dec.build_pipeline_report(results, alpha=0.05, policy=strict)
    assert report.decision.retraining_recommended is False
    assert report.decision.criteria["high_severity_drift_count"] == 2


def _model(prediction: FeatureDriftResult, imputed: dict[str, int] | None = None) -> dec.ModelMonitoring:
    return dec.ModelMonitoring(
        model_version="test-model",
        trained_at="2026-01-01T00:00:00+00:00",
        algorithm="LogisticRegression",
        training_data_sha256="0" * 64,
        lineage_verified=True,
        holdout_metrics={"roc_auc": 0.8},
        imputation_values={"temperature": 25.0},
        inference_rows=1000,
        imputed_counts=imputed or {},
        reference_positive_rate=0.15,
        current_positive_rate=0.90,
        reference_mean_score=0.27,
        current_mean_score=0.84,
        prediction_drift=prediction,
    )


def test_high_prediction_drift_alone_triggers_retraining() -> None:
    results = [_result(f"f{i}", STATUS_NO_DRIFT, SEVERITY_NONE) for i in range(5)]
    model = _model(_result("rain_probability", STATUS_DRIFT, SEVERITY_HIGH))
    report = dec.build_pipeline_report(results, alpha=0.05, model=model)
    assert report.decision.retraining_recommended is True
    assert any("Drift de predição de severidade alta" in r for r in report.decision.reasons)


def test_prediction_trigger_can_be_disabled() -> None:
    results = [_result(f"f{i}", STATUS_NO_DRIFT, SEVERITY_NONE) for i in range(5)]
    model = _model(_result("rain_probability", STATUS_DRIFT, SEVERITY_HIGH))
    policy = RetrainingPolicy(prediction_drift_triggers_retraining=False)
    report = dec.build_pipeline_report(results, alpha=0.05, model=model, policy=policy)
    assert report.decision.retraining_recommended is False
    assert any("abaixo do limiar" in r for r in report.decision.reasons)


def test_imputation_is_reported_as_reason() -> None:
    results = [_result("temperature", STATUS_NO_DRIFT, SEVERITY_NONE, missing_pct=5.0)]
    model = _model(_result("rain_probability", STATUS_NO_DRIFT, SEVERITY_NONE), {"temperature": 50})
    decision = dec.build_pipeline_report(results, alpha=0.05, model=model).decision
    assert any("50 valores ausentes de temperature" in r for r in decision.reasons)


def test_generated_data_decision(pipeline_report) -> None:
    summary, decision = pipeline_report.summary, pipeline_report.decision
    assert summary.total_features == 5
    assert summary.drifted_features == 4
    assert summary.drifted_pct == 80.0
    assert decision.retraining_recommended is True
    assert decision.action == dec.ACTION_RETRAIN_AFTER_REVIEW
    temperature = next(q for q in pipeline_report.data_quality if q.feature == "temperature")
    assert temperature.status == dec.QUALITY_WARNING
    assert temperature.current_missing_count == 250
    prediction = pipeline_report.model.prediction_drift
    assert prediction.status == STATUS_DRIFT and prediction.severity == SEVERITY_HIGH
