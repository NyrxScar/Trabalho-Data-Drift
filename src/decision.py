"""Consolidation of drift results, data-quality assessment and retraining rule.

Three concepts are deliberately kept apart:

1. **Statistical drift** - a test rejected the hypothesis of equal
   distributions (see :mod:`src.detect_drift`).
2. **Data quality** - integrity problems such as missing values or unknown
   categories, assessed here against configurable thresholds.
3. **Retraining** - an operational recommendation derived from explicit
   criteria (:class:`src.config.RetrainingPolicy`). The pipeline only raises
   the recommendation; it never retrains a model by itself.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src import config
from src.detect_drift import (
    SEVERITY_HIGH,
    STATUS_DRIFT,
    STATUS_NOT_TESTED,
    FeatureDriftResult,
)

logger = logging.getLogger(__name__)

QUALITY_OK = "ok"
QUALITY_WARNING = "warning"
QUALITY_CRITICAL = "critical"

ALERT_NONE = "none"
ALERT_WARNING = "warning"
ALERT_CRITICAL = "critical"

SEVERITY_LABELS_PT = {"none": "nenhuma", "low": "baixa", "moderate": "moderada", "high": "alta"}

ACTION_NO_ACTION = "NO_ACTION"
ACTION_MONITOR = "MONITOR"
ACTION_INVESTIGATE_DATA = "INVESTIGATE_DATA_QUALITY"
ACTION_RETRAIN = "RETRAIN_RECOMMENDED"
ACTION_RETRAIN_AFTER_REVIEW = "RETRAIN_RECOMMENDED_AFTER_DATA_QUALITY_REVIEW"
ACTION_FIX_THEN_RETRAIN = "FIX_DATA_PIPELINE_BEFORE_RETRAINING"


@dataclass
class DataQualityResult:
    """Data-quality verdict for one feature (production vs. thresholds)."""

    feature: str
    reference_missing_count: int
    reference_missing_pct: float
    current_missing_count: int
    current_missing_pct: float
    unexpected_categories: list[str]
    status: str
    issues: list[str] = field(default_factory=list)


@dataclass
class DriftSummary:
    """Aggregate view of the drift results."""

    total_features: int
    tested_features: int
    drifted_features: int
    drifted_pct: float
    drifted_feature_names: list[str]
    high_severity_feature_names: list[str]
    not_tested_feature_names: list[str]


@dataclass
class RetrainingDecision:
    """Operational recommendation and the reasons supporting it."""

    retraining_recommended: bool
    action: str
    alert_level: str
    reasons: list[str]
    criteria: dict[str, float | int]


@dataclass
class ModelMonitoring:
    """Facts about the production model collected during inference."""

    model_version: str
    trained_at: str
    algorithm: str
    training_data_sha256: str
    lineage_verified: bool
    holdout_metrics: dict[str, float]
    imputation_values: dict[str, float]
    inference_rows: int
    imputed_counts: dict[str, int]
    reference_positive_rate: float
    current_positive_rate: float
    reference_mean_score: float
    current_mean_score: float
    prediction_drift: FeatureDriftResult


@dataclass
class PipelineReport:
    """Everything produced by one monitoring run, consumed by the reports."""

    run_id: str
    timestamp: str
    alpha: float
    seed: int | None
    features: list[FeatureDriftResult]
    data_quality: list[DataQualityResult]
    summary: DriftSummary
    decision: RetrainingDecision
    model: ModelMonitoring | None = None
    data_generated_this_run: bool = False
    model_trained_this_run: bool = False


def assess_data_quality(
    results: list[FeatureDriftResult],
    policy: config.RetrainingPolicy = config.DEFAULT_RETRAINING_POLICY,
) -> list[DataQualityResult]:
    """Classify each feature's integrity based on production missing rates.

    * ``critical`` - missing % >= ``policy.missing_critical_pct``;
    * ``warning``  - missing % >= ``policy.missing_warning_pct`` or unknown
      categories were observed;
    * ``ok``       - otherwise.
    """
    assessments = []
    for result in results:
        missing = result.missing
        unexpected = list(result.details.get("unexpected_categories", []))
        issues: list[str] = []
        status = QUALITY_OK
        if missing.current_pct >= policy.missing_critical_pct:
            status = QUALITY_CRITICAL
            issues.append(
                f"{missing.current_pct:.2f}% de valores ausentes em produção "
                f"(limiar crítico {policy.missing_critical_pct}%)."
            )
        elif missing.current_pct >= policy.missing_warning_pct:
            status = QUALITY_WARNING
            issues.append(
                f"{missing.current_pct:.2f}% de valores ausentes em produção "
                f"(limiar de alerta {policy.missing_warning_pct}%; "
                f"referência {missing.reference_pct:.2f}%)."
            )
        if unexpected:
            status = QUALITY_CRITICAL if status == QUALITY_CRITICAL else QUALITY_WARNING
            issues.append(f"categorias inesperadas observadas: {unexpected}.")
        assessments.append(
            DataQualityResult(
                feature=result.feature,
                reference_missing_count=missing.reference_count,
                reference_missing_pct=missing.reference_pct,
                current_missing_count=missing.current_count,
                current_missing_pct=missing.current_pct,
                unexpected_categories=unexpected,
                status=status,
                issues=issues,
            )
        )
    return assessments


def summarize_drift(results: list[FeatureDriftResult]) -> DriftSummary:
    """Count drifted, high-severity and untested features."""
    drifted = [r.feature for r in results if r.status == STATUS_DRIFT]
    not_tested = [r.feature for r in results if r.status == STATUS_NOT_TESTED]
    high = [r.feature for r in results if r.status == STATUS_DRIFT and r.severity == SEVERITY_HIGH]
    total = len(results)
    return DriftSummary(
        total_features=total,
        tested_features=total - len(not_tested),
        drifted_features=len(drifted),
        drifted_pct=100.0 * len(drifted) / total if total else 0.0,
        drifted_feature_names=drifted,
        high_severity_feature_names=high,
        not_tested_feature_names=not_tested,
    )


def decide_retraining(
    summary: DriftSummary,
    quality: list[DataQualityResult],
    policy: config.RetrainingPolicy = config.DEFAULT_RETRAINING_POLICY,
    model: ModelMonitoring | None = None,
) -> RetrainingDecision:
    """Apply the documented retraining rule.

    Retraining is recommended if ``len(high severity drift) >=
    policy.high_severity_drift_count`` **or** ``drifted share >=
    policy.drifted_share_threshold`` **or** (when enabled) the model output
    shows high-severity prediction drift. Data-quality problems never trigger
    retraining alone, but they qualify the recommended action.
    """
    reasons: list[str] = []
    drifted_share = summary.drifted_features / summary.total_features if summary.total_features else 0.0

    high_trigger = len(summary.high_severity_feature_names) >= policy.high_severity_drift_count
    share_trigger = drifted_share >= policy.drifted_share_threshold
    if high_trigger:
        reasons.append(
            f"{len(summary.high_severity_feature_names)} variável(is) com drift de severidade alta "
            f"({', '.join(summary.high_severity_feature_names)}); "
            f"limiar: {policy.high_severity_drift_count}."
        )
    if share_trigger:
        reasons.append(
            f"{summary.drifted_pct:.1f}% das variáveis apresentam drift "
            f"(limiar: {policy.drifted_share_threshold:.0%})."
        )
    prediction_trigger = _prediction_trigger(model, policy, reasons)
    retrain = high_trigger or share_trigger or prediction_trigger
    if summary.drifted_features and not retrain:
        reasons.append(
            f"Drift detectado em {', '.join(summary.drifted_feature_names)}, porém abaixo "
            "dos limiares de retreinamento; manter monitoramento."
        )

    critical = [q.feature for q in quality if q.status == QUALITY_CRITICAL]
    warning = [q.feature for q in quality if q.status == QUALITY_WARNING]
    for item in quality:
        reasons.extend(f"Data Quality ({item.feature}): {issue}" for issue in item.issues)
    if summary.not_tested_feature_names:
        reasons.append(
            "Variáveis não testadas (ponto cego do monitoramento): "
            f"{', '.join(summary.not_tested_feature_names)}."
        )
    if model:
        reasons.extend(
            f"Inferência: {count} valores ausentes de {name} foram imputados com a "
            f"mediana de treino ({model.imputation_values[name]:.2f})."
            for name, count in model.imputed_counts.items()
        )

    action = _select_action(retrain, bool(critical), bool(warning), summary)
    alert = _select_alert(retrain, bool(critical), bool(warning), summary)
    if not reasons:
        reasons.append("Nenhum drift e nenhum problema de qualidade de dados detectado.")

    decision = RetrainingDecision(
        retraining_recommended=retrain,
        action=action,
        alert_level=alert,
        reasons=reasons,
        criteria={
            "high_severity_drift_count": policy.high_severity_drift_count,
            "drifted_share_threshold": policy.drifted_share_threshold,
            "prediction_drift_triggers_retraining": policy.prediction_drift_triggers_retraining,
            "missing_warning_pct": policy.missing_warning_pct,
            "missing_critical_pct": policy.missing_critical_pct,
            "ks_severity_moderate": config.KS_SEVERITY.moderate,
            "ks_severity_high": config.KS_SEVERITY.high,
            "cramers_v_severity_moderate": config.CRAMERS_V_SEVERITY.moderate,
            "cramers_v_severity_high": config.CRAMERS_V_SEVERITY.high,
        },
    )
    logger.info("Decision: action=%s, alert=%s, retrain=%s", action, alert, retrain)
    return decision


def _prediction_trigger(
    model: ModelMonitoring | None, policy: config.RetrainingPolicy, reasons: list[str]
) -> bool:
    """Evaluate the prediction-drift criterion and append its reason."""
    if model is None or model.prediction_drift.status != STATUS_DRIFT:
        return False
    drift = model.prediction_drift
    description = (
        f"as predições do modelo mudaram: taxa prevista de '{config.TARGET_COLUMN}' "
        f"{model.reference_positive_rate:.1%} → {model.current_positive_rate:.1%} "
        f"(KS D = {drift.statistic:.4f}, severidade {SEVERITY_LABELS_PT[drift.severity]})"
    )
    triggered = policy.prediction_drift_triggers_retraining and drift.severity == SEVERITY_HIGH
    if triggered:
        reasons.append(f"Drift de predição de severidade alta: {description}.")
    else:
        reasons.append(f"Drift de predição abaixo do limiar de retreinamento: {description}.")
    return triggered


def _select_action(
    retrain: bool, critical: bool, warning: bool, summary: DriftSummary
) -> str:
    if retrain and critical:
        return ACTION_FIX_THEN_RETRAIN
    if retrain and warning:
        return ACTION_RETRAIN_AFTER_REVIEW
    if retrain:
        return ACTION_RETRAIN
    if critical or warning or summary.not_tested_feature_names:
        return ACTION_INVESTIGATE_DATA
    if summary.drifted_features:
        return ACTION_MONITOR
    return ACTION_NO_ACTION


def _select_alert(
    retrain: bool, critical: bool, warning: bool, summary: DriftSummary
) -> str:
    if retrain or critical:
        return ALERT_CRITICAL
    if warning or summary.drifted_features or summary.not_tested_feature_names:
        return ALERT_WARNING
    return ALERT_NONE


def new_run_id(timestamp: datetime) -> str:
    """Build a unique, sortable run identifier."""
    return f"run-{timestamp:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def build_pipeline_report(
    results: list[FeatureDriftResult],
    alpha: float,
    seed: int | None = None,
    policy: config.RetrainingPolicy = config.DEFAULT_RETRAINING_POLICY,
    model: ModelMonitoring | None = None,
    data_generated_this_run: bool = False,
    model_trained_this_run: bool = False,
) -> PipelineReport:
    """Consolidate test results, quality checks and decision into one object."""
    now = datetime.now(timezone.utc)
    quality = assess_data_quality(results, policy)
    summary = summarize_drift(results)
    return PipelineReport(
        run_id=new_run_id(now),
        timestamp=now.isoformat(timespec="seconds"),
        alpha=alpha,
        seed=seed,
        features=results,
        data_quality=quality,
        summary=summary,
        decision=decide_retraining(summary, quality, policy, model),
        model=model,
        data_generated_this_run=data_generated_this_run,
        model_trained_this_run=model_trained_this_run,
    )
