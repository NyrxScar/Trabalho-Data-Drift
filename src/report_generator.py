"""Report generation from the statistics computed by the pipeline.

The monitoring stage writes the audit log ``data/drift_summary.json``. The
HTML reports are then built by *reading* that JSON back (plus the CSV files
for the charts), so reporting is decoupled from computation and every number
shown is exactly the one that was logged. Statistical tests are never
re-executed here.

Artefacts:

* ``data/drift_summary.json`` - machine-readable audit log;
* ``reports/exec_summary.html`` - executive summary (HTML + CSS);
* ``reports/drift_report.html`` - interactive Plotly report.

Usage (regenerate the HTML reports from an existing audit log)::

    python -m src.report_generator
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config
from src.artifacts import read_json, to_jsonable, write_json
from src.decision import (
    ACTION_FIX_THEN_RETRAIN,
    ACTION_INVESTIGATE_DATA,
    ACTION_MONITOR,
    ACTION_NO_ACTION,
    ACTION_RETRAIN,
    ACTION_RETRAIN_AFTER_REVIEW,
    QUALITY_OK,
    DataQualityResult,
    DriftSummary,
    ModelMonitoring,
    PipelineReport,
    RetrainingDecision,
)
from src.detect_drift import (
    CHI2_METHOD_MONTE_CARLO,
    CHI2_TEST_NAME,
    KS_TEST_NAME,
    STATUS_DRIFT,
    STATUS_NO_DRIFT,
    STATUS_NOT_TESTED,
    FeatureDriftResult,
    MissingValueStats,
    load_dataset,
)

logger = logging.getLogger(__name__)

JSON_SCHEMA_VERSION = "2.0"
HISTOGRAM_BINS = 40
SMALLEST_REPORTABLE_P = 1e-300
NOT_AVAILABLE = "n/d"

# Colors validated with the dataviz palette checker (reference vs production).
COLOR_REFERENCE = "#2a78d6"
COLOR_CURRENT = "#eb6834"
COLOR_SURFACE = "#fcfcfb"
STATUS_COLORS = {
    STATUS_DRIFT: "#d03b3b",
    STATUS_NO_DRIFT: "#0ca30c",
    STATUS_NOT_TESTED: "#fab219",
}

STATUS_LABELS = {
    STATUS_DRIFT: "🚨 DRIFT",
    STATUS_NO_DRIFT: "✅ Estável",
    STATUS_NOT_TESTED: "⚠️ Não testado",
}
SEVERITY_LABELS = {"none": "—", "low": "Baixa", "moderate": "Moderada", "high": "Alta"}
QUALITY_LABELS = {"ok": "✅ OK", "warning": "⚠️ Alerta", "critical": "🚨 Crítico"}
TEST_SHORT_LABELS = {KS_TEST_NAME: "KS", CHI2_TEST_NAME: "Qui-Quadrado"}
TYPE_LABELS = {"numeric": "Numérica", "categorical": "Categórica", "prediction": "Predição"}
EFFECT_LABELS = {"ks_statistic": "KS D", "cramers_v": "V de Cramér"}
ALERT_LABELS = {"none": "Sem alerta", "warning": "Alerta", "critical": "Alerta crítico"}
ALERT_ICONS = {"none": "✅", "warning": "⚠️", "critical": "🚨"}
METRIC_LABELS = {
    "roc_auc": "ROC AUC",
    "accuracy": "Acurácia",
    "precision": "Precisão",
    "recall": "Recall",
    "f1": "F1",
    "observed_positive_rate": "Taxa real de chuva",
    "predicted_positive_rate": "Taxa prevista de chuva",
}
ACTION_DESCRIPTIONS = {
    ACTION_NO_ACTION: "Nenhuma ação necessária. Manter o monitoramento periódico.",
    ACTION_MONITOR: "Drift de baixa relevância. Manter o monitoramento e reavaliar na próxima janela.",
    ACTION_INVESTIGATE_DATA: "Investigar a qualidade dos dados de entrada antes de qualquer outra ação.",
    ACTION_RETRAIN: "Retreinamento recomendado com dados recentes de produção.",
    ACTION_RETRAIN_AFTER_REVIEW: (
        "Retreinamento recomendado, após revisão da qualidade dos dados "
        "(tratar ou corrigir os valores ausentes antes de montar o novo conjunto de treino)."
    ),
    ACTION_FIX_THEN_RETRAIN: (
        "Corrigir o pipeline de dados antes de retreinar: a qualidade atual "
        "comprometeria o novo modelo."
    ),
}


# --------------------------------------------------------------------------- #
# Formatting helpers (shared by every report, so values always match)
# --------------------------------------------------------------------------- #
def format_p_value(p_value: float | None) -> str:
    """Human-readable p-value; underflowed zeros are shown as an upper bound."""
    if p_value is None:
        return NOT_AVAILABLE
    if p_value < SMALLEST_REPORTABLE_P:
        return f"< {SMALLEST_REPORTABLE_P:.0e}"
    return f"{p_value:.3e}" if p_value < 1e-3 else f"{p_value:.4f}"


def p_value_clause(p_value: float | None, symbol: str = "p") -> str:
    """Render ``p = x`` or ``p < bound`` without doubling the operator."""
    text = format_p_value(p_value)
    return f"{symbol} {text}" if text.startswith("<") else f"{symbol} = {text}"


def format_statistic(value: float | None) -> str:
    """Format a test statistic or effect size with four decimals."""
    return NOT_AVAILABLE if value is None else f"{value:.4f}"


def format_pct(value: float) -> str:
    """Format a percentage with two decimals."""
    return f"{value:.2f}%"


# --------------------------------------------------------------------------- #
# JSON audit log: report -> payload -> report
# --------------------------------------------------------------------------- #
def _feature_payload(result: FeatureDriftResult, quality_status: str | None) -> dict[str, Any]:
    return {
        "feature": result.feature,
        "feature_type": result.feature_type,
        "test": result.test_name,
        "statistic": result.statistic,
        "p_value": result.p_value,
        "alpha": result.alpha,
        "drift_detected": result.drift_detected,
        "status": result.status,
        "effect_size_metric": result.effect_size_name,
        "effect_size": result.effect_size,
        "severity": result.severity,
        "reference_count": result.reference_count,
        "current_count": result.current_count,
        "reference_valid_count": result.reference_valid_count,
        "current_valid_count": result.current_valid_count,
        "has_missing_values": (
            result.missing.reference_count + result.missing.current_count
        ) > 0,
        "missing_values": asdict(result.missing),
        "data_quality_status": quality_status,
        "interpretation": result.interpretation,
        "error": result.error,
        "details": result.details,
    }


def _model_payload(model: ModelMonitoring | None) -> dict[str, Any] | None:
    if model is None:
        return None
    payload = asdict(model)
    payload["prediction_drift"] = _feature_payload(model.prediction_drift, None)
    return payload


def build_summary_payload(report: PipelineReport) -> dict[str, Any]:
    """Assemble the audit-log dictionary from a pipeline report."""
    quality_by_feature = {q.feature: q for q in report.data_quality}
    payload = {
        "schema_version": JSON_SCHEMA_VERSION,
        "run": {
            "run_id": report.run_id,
            "timestamp": report.timestamp,
            "seed": report.seed,
            "alpha": report.alpha,
            "data_generated_this_run": report.data_generated_this_run,
            "model_trained_this_run": report.model_trained_this_run,
        },
        "summary": {
            "total_features_analyzed": report.summary.total_features,
            "tested_features": report.summary.tested_features,
            "features_with_drift": report.summary.drifted_features,
            "drifted_features_pct": report.summary.drifted_pct,
            "drifted_features": report.summary.drifted_feature_names,
            "high_severity_features": report.summary.high_severity_feature_names,
            "not_tested_features": report.summary.not_tested_feature_names,
        },
        "features": [
            _feature_payload(r, quality_by_feature[r.feature].status) for r in report.features
        ],
        "data_quality": {
            "overall_status": _overall_quality(report.data_quality),
            "features": [asdict(q) for q in report.data_quality],
        },
        "model": _model_payload(report.model),
        "retraining_decision": asdict(report.decision),
    }
    return to_jsonable(payload)


def _feature_from_payload(entry: dict[str, Any]) -> FeatureDriftResult:
    return FeatureDriftResult(
        feature=entry["feature"],
        feature_type=entry["feature_type"],
        test_name=entry["test"],
        alpha=entry["alpha"],
        reference_count=entry["reference_count"],
        current_count=entry["current_count"],
        reference_valid_count=entry["reference_valid_count"],
        current_valid_count=entry["current_valid_count"],
        missing=MissingValueStats(**entry["missing_values"]),
        statistic=entry["statistic"],
        p_value=entry["p_value"],
        drift_detected=entry["drift_detected"],
        status=entry["status"],
        effect_size_name=entry["effect_size_metric"],
        effect_size=entry["effect_size"],
        severity=entry["severity"],
        interpretation=entry["interpretation"],
        error=entry["error"],
        details=entry["details"],
    )


def report_from_payload(payload: dict[str, Any]) -> PipelineReport:
    """Rebuild a :class:`PipelineReport` from the audit-log dictionary.

    Raises:
        ValueError: If the payload has an unsupported schema or misses keys.
    """
    if payload.get("schema_version") != JSON_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported audit-log schema {payload.get('schema_version')!r}; "
            f"expected {JSON_SCHEMA_VERSION!r}."
        )
    try:
        run, summary = payload["run"], payload["summary"]
        model_entry = payload["model"]
        model = None
        if model_entry is not None:
            model = ModelMonitoring(
                **{**model_entry, "prediction_drift": _feature_from_payload(model_entry["prediction_drift"])}
            )
        return PipelineReport(
            run_id=run["run_id"],
            timestamp=run["timestamp"],
            alpha=run["alpha"],
            seed=run["seed"],
            features=[_feature_from_payload(f) for f in payload["features"]],
            data_quality=[DataQualityResult(**q) for q in payload["data_quality"]["features"]],
            summary=DriftSummary(
                total_features=summary["total_features_analyzed"],
                tested_features=summary["tested_features"],
                drifted_features=summary["features_with_drift"],
                drifted_pct=summary["drifted_features_pct"],
                drifted_feature_names=summary["drifted_features"],
                high_severity_feature_names=summary["high_severity_features"],
                not_tested_feature_names=summary["not_tested_features"],
            ),
            decision=RetrainingDecision(**payload["retraining_decision"]),
            model=model,
            data_generated_this_run=run["data_generated_this_run"],
            model_trained_this_run=run["model_trained_this_run"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed audit log: {exc}") from exc


def write_json_summary(report: PipelineReport, path: Path) -> Path:
    """Write the audit log as strict JSON (``allow_nan=False``)."""
    write_json(build_summary_payload(report), path)
    _log_written(path)
    return path


def load_summary(path: Path) -> PipelineReport:
    """Read the audit log and rebuild the report it describes."""
    return report_from_payload(read_json(path))


def _overall_quality(quality: list[DataQualityResult]) -> str:
    order = ["ok", "warning", "critical"]
    return max((q.status for q in quality), key=order.index, default=QUALITY_OK)


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Failed to write report to {path}.")
    _log_written(path)
    return path


def _log_written(path: Path) -> None:
    logger.info("Report written: %s (%.1f KB).", path, path.stat().st_size / 1024)


# --------------------------------------------------------------------------- #
# Executive summary (HTML)
# --------------------------------------------------------------------------- #
EXEC_CSS = """
:root {
  color-scheme: light;
  --bg: #f5f5f3; --surface: #fcfcfb; --border: #dcdbd6;
  --text: #0b0b0b; --text-2: #52514e; --muted: #75746f;
  --good: #0a7d0a; --warning: #9a6b00; --critical: #c03030;
  --good-bg: #e7f5e7; --warning-bg: #fdf3dc; --critical-bg: #fbe9e9;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --border: #3a3936;
    --text: #f0efec; --text-2: #c3c2b7; --muted: #9a998f;
    --good: #3cc23c; --warning: #fab219; --critical: #e66767;
    --good-bg: #13301a; --warning-bg: #3a2e10; --critical-bg: #3b1a1a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --border: #3a3936;
  --text: #f0efec; --text-2: #c3c2b7; --muted: #9a998f;
  --good: #3cc23c; --warning: #fab219; --critical: #e66767;
  --good-bg: #13301a; --warning-bg: #3a2e10; --critical-bg: #3b1a1a;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 1.7rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 36px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.meta { color: var(--text-2); font-size: .9rem; }
.meta code { font-size: .85rem; }
.banner { margin-top: 20px; padding: 16px 18px; border-radius: 10px; border-left: 6px solid; background: var(--surface); }
.banner.critical { border-color: var(--critical); background: var(--critical-bg); }
.banner.warning { border-color: var(--warning); background: var(--warning-bg); }
.banner.none { border-color: var(--good); background: var(--good-bg); }
.banner strong { font-size: 1.05rem; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 16px; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.kpi .value { font-size: 1.9rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.kpi .label { color: var(--text-2); font-size: .85rem; }
.callout { margin-top: 16px; padding: 14px 16px; border-radius: 10px; background: var(--warning-bg); border: 1px solid var(--warning); }
.table-wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
table { width: 100%; border-collapse: collapse; font-size: .9rem; }
th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--text-2); font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .03em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.drift td { background: var(--critical-bg); }
tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: .8rem; font-weight: 600; border: 1px solid; }
.badge.drift, .badge.critical { color: var(--critical); border-color: var(--critical); }
.badge.no_drift, .badge.ok { color: var(--good); border-color: var(--good); }
.badge.not_tested, .badge.warning { color: var(--warning); border-color: var(--warning); }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.card h3 { margin: 0 0 8px; font-size: .95rem; }
dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 16px; margin: 0; font-size: .9rem; }
dt { color: var(--text-2); }
dd { margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
ul { padding-left: 20px; }
li { margin: 6px 0; }
.small { color: var(--muted); font-size: .85rem; }
footer { margin-top: 40px; color: var(--muted); font-size: .8rem; }
"""


def _badge(css_class: str, text: str) -> str:
    return f'<span class="badge {escape(css_class)}">{escape(text)}</span>'


def _kpi(value: str, label: str) -> str:
    return (
        f'<div class="kpi"><div class="value">{escape(value)}</div>'
        f'<div class="label">{escape(label)}</div></div>'
    )


def _definition_list(items: list[tuple[str, str]]) -> str:
    rows = "".join(f"<dt>{escape(k)}</dt><dd>{v}</dd>" for k, v in items)
    return f"<dl>{rows}</dl>"


def _results_table(report: PipelineReport) -> str:
    quality = {q.feature: q for q in report.data_quality}
    rows = []
    for r in report.features:
        q = quality[r.feature]
        rows.append(
            f'<tr class="{escape(r.status)}">'
            f"<td><strong>{escape(r.feature)}</strong></td>"
            f"<td>{TYPE_LABELS.get(r.feature_type, r.feature_type)}</td>"
            f"<td>{escape(TEST_SHORT_LABELS.get(r.test_name, r.test_name))}</td>"
            f'<td class="num">{format_statistic(r.statistic)}</td>'
            f'<td class="num">{escape(format_p_value(r.p_value))}</td>'
            f'<td class="num">{EFFECT_LABELS.get(r.effect_size_name or "", "")} '
            f"{format_statistic(r.effect_size)}</td>"
            f"<td>{SEVERITY_LABELS.get(r.severity, r.severity)}</td>"
            f"<td>{_badge(r.status, STATUS_LABELS[r.status])}</td>"
            f'<td class="num">{q.current_missing_count} ({format_pct(q.current_missing_pct)})</td>'
            f"<td>{_badge(q.status, QUALITY_LABELS[q.status])}</td>"
            "</tr>"
        )
    header = (
        "<tr><th>Variável</th><th>Tipo</th><th>Teste</th><th>Estatística</th>"
        "<th>P-value</th><th>Efeito</th><th>Severidade</th><th>Status</th>"
        "<th>NaN (prod.)</th><th>Data Quality</th></tr>"
    )
    return f'<div class="table-wrap"><table>{header}{"".join(rows)}</table></div>'


def _missing_callouts(report: PipelineReport) -> str:
    imputed = report.model.imputed_counts if report.model else {}
    blocks = []
    for q in report.data_quality:
        if q.current_missing_count == 0:
            continue
        imputation = ""
        if q.feature in imputed:
            imputation = (
                f" Na inferência, esses {imputed[q.feature]} registros foram imputados com a "
                f"mediana de treino ({report.model.imputation_values[q.feature]:.2f})."
            )
        blocks.append(
            '<div class="callout"><strong>⚠️ Falha de Data Quality: valores ausentes (NaN) em '
            f"<code>{escape(q.feature)}</code></strong>: {q.current_missing_count} registros "
            f"({format_pct(q.current_missing_pct)}) estão vazios no conjunto de produção, contra "
            f"{q.reference_missing_count} ({format_pct(q.reference_missing_pct)}) na referência. "
            f"Status de qualidade: <strong>{QUALITY_LABELS[q.status]}</strong>. "
            "O teste de drift desta variável foi calculado apenas sobre os valores válidos; "
            f"os ausentes são tratados como um problema de integridade, não como drift.{imputation}</div>"
        )
    return "".join(blocks)


def describe_feature(result: FeatureDriftResult) -> str:
    """Build a data-driven diagnostic sentence for one feature (pt-BR)."""
    name = f"<strong>{escape(result.feature)}</strong>"
    if result.status == STATUS_NOT_TESTED:
        return f"{name}: teste não executado: {escape(result.error or '')}"

    stat_text = (
        f"{escape(result.test_name)}: estatística = {format_statistic(result.statistic)}, "
        f"{escape(p_value_clause(result.p_value, 'p-value'))}"
    )
    d = result.details
    if result.feature_type == "categorical":
        parts = [
            f"{escape(cat)} {d['reference_proportions'][cat]:.1%} → "
            f"{d['current_proportions'][cat]:.1%} "
            f"({(d['current_proportions'][cat] - d['reference_proportions'][cat]) * 100:+.1f} p.p.)"
            for cat in d["categories"]
        ]
        method = (
            "p-value por permutação Monte Carlo (condições de Cochran não atendidas)"
            if d.get("p_value_method") == CHI2_METHOD_MONTE_CARLO
            else "aproximação assintótica válida (condições de Cochran atendidas)"
        )
        change = f"proporções {'; '.join(parts)}; graus de liberdade = {d['degrees_of_freedom']}; {method}"
    else:
        change = (
            f"média {d['reference_mean']:.2f} → {d['current_mean']:.2f} "
            f"({d['current_mean'] - d['reference_mean']:+.2f}); "
            f"desvio padrão {d['reference_std']:.2f} → {d['current_std']:.2f}; "
            f"mediana {d['reference_median']:.2f} → {d['current_median']:.2f}"
        )

    effect = (
        f"{EFFECT_LABELS.get(result.effect_size_name or '', '')} = "
        f"{format_statistic(result.effect_size)}"
    )
    if result.status == STATUS_DRIFT:
        verdict = f"drift detectado, severidade <strong>{SEVERITY_LABELS[result.severity].lower()}</strong>"
    else:
        verdict = (
            "estável: sem evidência estatística de drift ao nível alpha configurado "
            "(o que não comprova que as distribuições sejam idênticas)"
        )
    return f"{name}: {verdict}. {change}. {stat_text}; {effect}."


def _impact_paragraph(report: PipelineReport) -> str:
    drifted = report.summary.drifted_feature_names
    missing = [q.feature for q in report.data_quality if q.current_missing_count > 0]
    sentences = []
    if drifted:
        sentences.append(
            f"As variáveis {', '.join(drifted)} chegam em produção com distribuição diferente "
            "da usada no treinamento. O modelo passa a receber entradas em regiões pouco "
            "representadas no treino, o que tende a degradar a acurácia e a calibração."
        )
    if report.model:
        m = report.model
        sentences.append(
            f"Efeito observado no modelo {m.model_version}: a taxa prevista de chuva passou de "
            f"{m.reference_positive_rate:.1%} (referência) para {m.current_positive_rate:.1%} "
            f"(produção) e o score médio de {m.reference_mean_score:.3f} para "
            f"{m.current_mean_score:.3f}. Como os rótulos de produção ainda não estão "
            "disponíveis, não é possível afirmar se essas previsões estão corretas; a "
            "confirmação depende do ground truth."
        )
    if missing:
        sentences.append(
            f"Os valores ausentes em {', '.join(missing)} são imputados com a mediana do "
            "período de treino, que não representa mais o padrão atual; isso enviesa as "
            "previsões desses registros e mascara a falha do sensor."
        )
    if not sentences:
        sentences.append("Não foram identificados sinais que indiquem risco para o modelo nesta execução.")
    return " ".join(sentences)


def _model_section(report: PipelineReport) -> str:
    m = report.model
    if m is None:
        return "<p>Nenhum modelo foi avaliado nesta execução.</p>"
    drift = m.prediction_drift
    artefact = _definition_list(
        [
            ("Versão", f"<code>{escape(m.model_version)}</code>"),
            ("Algoritmo", escape(m.algorithm)),
            ("Treinado em", f"{escape(m.trained_at)} (UTC)"),
            (
                "Treinado nesta execução",
                "Sim" if report.model_trained_this_run else "Não (artefato .joblib reutilizado)",
            ),
            (
                "Linhagem dos dados",
                "✅ verificada (SHA-256 da referência confere)"
                if m.lineage_verified
                else "🚨 divergente",
            ),
            ("SHA-256 do treino", f"<code>{escape(m.training_data_sha256[:16])}…</code>"),
        ]
    )
    metrics = _definition_list(
        [(METRIC_LABELS.get(k, k), f"{v:.4f}") for k, v in m.holdout_metrics.items()]
    )
    imputed = ", ".join(f"{k}: {v}" for k, v in m.imputed_counts.items()) or "nenhuma"
    behaviour = _definition_list(
        [
            ("Registros inferidos", str(m.inference_rows)),
            ("Imputações (NaN)", escape(imputed)),
            (
                "Taxa prevista de chuva",
                f"{m.reference_positive_rate:.1%} → {m.current_positive_rate:.1%}",
            ),
            ("Score médio", f"{m.reference_mean_score:.3f} → {m.current_mean_score:.3f}"),
            (
                "Drift de predição",
                f"{_badge(drift.status, STATUS_LABELS[drift.status])} KS D = "
                f"{format_statistic(drift.statistic)}, {escape(p_value_clause(drift.p_value))}, "
                f"severidade {SEVERITY_LABELS[drift.severity].lower()}",
            ),
        ]
    )
    return (
        '<div class="grid-2">'
        f'<div class="card"><h3>Artefato em produção</h3>{artefact}</div>'
        f'<div class="card"><h3>Métricas no holdout (treino)</h3>{metrics}</div>'
        f'<div class="card"><h3>Comportamento em produção</h3>{behaviour}</div>'
        "</div>"
        '<p class="small">O pipeline de monitoramento apenas carrega o modelo salvo e executa a '
        "inferência; o treino é uma etapa separada (<code>python -m src.train_model</code>).</p>"
    )


def render_exec_summary(report: PipelineReport) -> str:
    """Render the executive summary HTML entirely from ``report``."""
    summary, decision = report.summary, report.decision
    quality_issues = sum(1 for q in report.data_quality if q.status != QUALITY_OK)
    diagnostics = "".join(f"<li>{describe_feature(r)}</li>" for r in report.features)
    reasons = "".join(f"<li>{escape(reason)}</li>" for reason in decision.reasons)
    criteria = "".join(
        f"<li><code>{escape(key)}</code> = {escape(str(value))}</li>"
        for key, value in decision.criteria.items()
    )
    retrain_text = "SIM" if decision.retraining_recommended else "NÃO"
    health_status = (
        "🚨 Saúde dos dados comprometida"
        if summary.drifted_features or quality_issues
        else "✅ Dados saudáveis"
    )
    health = (
        f"{summary.drifted_features} de {summary.total_features} variáveis "
        f"({summary.drifted_pct:.1f}%) apresentam drift estatístico; "
        f"{quality_issues} variável(is) com falha de qualidade de dados."
    )

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Resumo Executivo de Drift</title>
<style>{EXEC_CSS}</style>
</head>
<body>
<main>
<h1>Relatório Executivo: Monitoramento de Data Drift</h1>
<div class="meta">Execução <code>{escape(report.run_id)}</code> · Análise em {escape(report.timestamp)} (UTC)
 · alpha = {report.alpha} · seed dos dados = {escape(str(report.seed))}</div>

<div class="banner {escape(decision.alert_level)}">
  <strong>{ALERT_ICONS[decision.alert_level]} {ALERT_LABELS[decision.alert_level]}: retreinamento recomendado? {retrain_text}</strong><br>
  {escape(ACTION_DESCRIPTIONS[decision.action])}
  <div class="small">Ação: <code>{escape(decision.action)}</code></div>
</div>

<h2>Diagnóstico da saúde dos dados</h2>
<p><strong>{escape(health_status)}.</strong> {escape(health)}</p>
<div class="kpis">
  {_kpi(str(summary.total_features), "variáveis analisadas")}
  {_kpi(f"{summary.drifted_features} ({summary.drifted_pct:.0f}%)", "variáveis com drift")}
  {_kpi(str(len(summary.high_severity_feature_names)), "drift de severidade alta")}
  {_kpi(str(quality_issues), "variáveis com falha de qualidade")}
</div>
{_missing_callouts(report)}

<h2>Resultados consolidados</h2>
{_results_table(report)}
<p class="small">🚨 DRIFT: p-value &lt; alpha. ✅ Estável: sem evidência estatística de drift
(não comprova igualdade). Efeito: KS D para numéricas (distância máxima entre as distribuições
acumuladas) e V de Cramér para categóricas.</p>

<h2>Modelo em produção</h2>
{_model_section(report)}

<h2>Diagnóstico técnico</h2>
<ul>{diagnostics}</ul>
<p>{escape(_impact_paragraph(report))}</p>

<h2>Recomendação de ação</h2>
<p><strong>{escape(ACTION_DESCRIPTIONS[decision.action])}</strong></p>
<p>Motivos que sustentam a decisão:</p>
<ul>{reasons}</ul>
<p class="small">O pipeline apenas recomenda; o retreinamento não é executado automaticamente.
Critérios aplicados:</p>
<ul class="small">{criteria}</ul>

<footer>Gerado automaticamente por <code>src/report_generator.py</code> a partir de
<code>data/drift_summary.json</code> (execução <code>{escape(report.run_id)}</code>). Drift
estatístico, qualidade de dados e decisão de retreinamento são avaliados separadamente.</footer>
</main>
</body>
</html>
"""


def write_exec_summary(report: PipelineReport, path: Path) -> Path:
    """Write the executive summary HTML file."""
    return _write_text(render_exec_summary(report), path)


# --------------------------------------------------------------------------- #
# Interactive Plotly report
# --------------------------------------------------------------------------- #
def _test_title(result: FeatureDriftResult, statistic_label: str) -> str:
    return (
        f"{result.feature}: {statistic_label}={format_statistic(result.statistic)}, "
        f"{p_value_clause(result.p_value)} ({STATUS_LABELS[result.status]})"
    )


def _add_histograms(
    fig: go.Figure,
    reference: pd.Series,
    current: pd.Series,
    row: int,
    col: int,
    show_legend: bool,
) -> None:
    ref_valid, cur_valid = reference.dropna(), current.dropna()
    edges = np.histogram_bin_edges(pd.concat([ref_valid, cur_valid]), bins=HISTOGRAM_BINS)
    bins = {"start": float(edges[0]), "end": float(edges[-1]), "size": float(edges[1] - edges[0])}
    for series, label, color in (
        (ref_valid, "Referência", COLOR_REFERENCE),
        (cur_valid, "Produção", COLOR_CURRENT),
    ):
        fig.add_trace(
            go.Histogram(
                x=series,
                name=label,
                legendgroup=label,
                showlegend=show_legend,
                xbins=bins,
                histnorm="probability density",
                marker={"color": color, "line": {"color": COLOR_SURFACE, "width": 1}},
                opacity=0.55,
                hovertemplate=f"{label}<br>%{{x}}<br>densidade=%{{y:.4f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )


def _add_paired_bars(
    fig: go.Figure,
    categories: list[str],
    reference_values: list[float],
    current_values: list[float],
    text_format: str,
    hover_format: str,
    row: int,
    col: int,
) -> None:
    for values, label, color in (
        (reference_values, "Referência", COLOR_REFERENCE),
        (current_values, "Produção", COLOR_CURRENT),
    ):
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                name=label,
                legendgroup=label,
                showlegend=False,
                offsetgroup=label,
                marker={"color": color, "line": {"color": COLOR_SURFACE, "width": 2}},
                text=[format(v, text_format) if v else "" for v in values],
                textposition="outside",
                hovertemplate=f"{label}<br>%{{x}}: %{{y:{hover_format}}}<extra></extra>",
            ),
            row=row,
            col=col,
        )


def build_drift_figure(
    report: PipelineReport,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_predictions: pd.DataFrame | None = None,
    current_predictions: pd.DataFrame | None = None,
) -> go.Figure:
    """Build the Plotly figure from the persisted data and the logged results."""
    numeric = [r for r in report.features if r.feature_type == "numeric"]
    categorical = next(r for r in report.features if r.feature_type == "categorical")
    show_model = (
        report.model is not None
        and reference_predictions is not None
        and current_predictions is not None
    )

    titles = [_test_title(r, "KS D") for r in numeric] + [
        _test_title(categorical, "χ²"),
        "Valores ausentes (NaN) por variável (%)",
        "Tamanho de efeito por variável (KS D / V de Cramér)",
        "Resultados estatísticos",
    ]
    specs = [[{}, {}], [{}, {}], [{}, {}], [{}, {"type": "table"}]]
    if show_model:
        titles += [
            _test_title(report.model.prediction_drift, "KS D"),
            "Taxa prevista de chuva (modelo em produção)",
        ]
        specs.append([{}, {}])
    rows = len(specs)
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
        specs=specs,
    )

    for index, result in enumerate(numeric):
        row, col = divmod(index, 2)
        _add_histograms(
            fig, reference[result.feature], current[result.feature], row + 1, col + 1, index == 0
        )
        fig.update_xaxes(title_text=result.feature, row=row + 1, col=col + 1)
        fig.update_yaxes(title_text="densidade", row=row + 1, col=col + 1)

    _add_category_bars(fig, categorical, row=3, col=1)
    _add_missing_bars(fig, report, row=3, col=2)
    _add_effect_bars(fig, report, row=4, col=1)
    _add_results_table(fig, report, row=4, col=2)
    if show_model:
        _add_prediction_charts(fig, report, reference_predictions, current_predictions, row=5)

    fig.update_layout(
        height=420 * rows,
        barmode="overlay",
        template="plotly_white",
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_SURFACE,
        font={"family": "system-ui, Segoe UI, Roboto, sans-serif", "size": 12, "color": "#0b0b0b"},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"t": 80, "l": 60, "r": 30, "b": 40},
    )
    fig.update_annotations(font_size=12)
    return fig


def _add_category_bars(fig: go.Figure, result: FeatureDriftResult, row: int, col: int) -> None:
    categories = result.details["categories"]
    _add_paired_bars(
        fig,
        categories,
        [result.details["reference_proportions"][c] for c in categories],
        [result.details["current_proportions"][c] for c in categories],
        ".0%",
        ".1%",
        row,
        col,
    )
    fig.update_yaxes(title_text="proporção", tickformat=".0%", range=[0, 0.75], row=row, col=col)


def _add_missing_bars(fig: go.Figure, report: PipelineReport, row: int, col: int) -> None:
    quality = report.data_quality
    _add_paired_bars(
        fig,
        [q.feature for q in quality],
        [q.reference_missing_pct for q in quality],
        [q.current_missing_pct for q in quality],
        ".2f",
        ".2f",
        row,
        col,
    )
    fig.add_hline(
        y=config.MISSING_WARNING_PCT,
        line_dash="dot",
        line_color="#75746f",
        annotation_text=f"limiar de alerta {config.MISSING_WARNING_PCT}%",
        annotation_font_color="#52514e",
        row=row,
        col=col,
    )
    top = max([q.current_missing_pct for q in quality] + [config.MISSING_WARNING_PCT])
    fig.update_yaxes(title_text="% ausentes", range=[0, top * 1.4], row=row, col=col)


def _add_effect_bars(fig: go.Figure, report: PipelineReport, row: int, col: int) -> None:
    tested = [r for r in report.features if r.effect_size is not None]
    fig.add_trace(
        go.Bar(
            x=[r.feature for r in tested],
            y=[r.effect_size for r in tested],
            showlegend=False,
            marker={
                "color": [STATUS_COLORS[r.status] for r in tested],
                "line": {"color": COLOR_SURFACE, "width": 2},
            },
            text=[
                f"{STATUS_LABELS[r.status]}<br>{format_statistic(r.effect_size)}" for r in tested
            ],
            textposition="outside",
            customdata=[
                [
                    EFFECT_LABELS[r.effect_size_name],
                    format_p_value(r.p_value),
                    SEVERITY_LABELS[r.severity],
                ]
                for r in tested
            ],
            hovertemplate=(
                "%{x}<br>%{customdata[0]} = %{y:.4f}<br>p-value = %{customdata[1]}"
                "<br>severidade: %{customdata[2]}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )
    top = max([r.effect_size for r in tested] + [config.CRAMERS_V_SEVERITY.high])
    fig.update_yaxes(title_text="tamanho de efeito", range=[0, top * 1.35], row=row, col=col)


def _add_results_table(fig: go.Figure, report: PipelineReport, row: int, col: int) -> None:
    quality = {q.feature: q for q in report.data_quality}
    results = report.features
    fig.add_trace(
        go.Table(
            header={
                "values": ["Variável", "Estatística", "P-value", "Status", "NaN (prod.)"],
                "fill_color": "#efeeea",
                "align": "left",
                "font": {"color": "#0b0b0b"},
            },
            cells={
                "values": [
                    [r.feature for r in results],
                    [format_statistic(r.statistic) for r in results],
                    [format_p_value(r.p_value) for r in results],
                    [STATUS_LABELS[r.status] for r in results],
                    [format_pct(quality[r.feature].current_missing_pct) for r in results],
                ],
                "align": "left",
                "fill_color": COLOR_SURFACE,
                "font": {"color": "#0b0b0b"},
                "height": 26,
            },
        ),
        row=row,
        col=col,
    )


def _add_prediction_charts(
    fig: go.Figure,
    report: PipelineReport,
    reference_predictions: pd.DataFrame,
    current_predictions: pd.DataFrame,
    row: int,
) -> None:
    score = config.PREDICTION_SCORE_COLUMN
    _add_histograms(
        fig, reference_predictions[score], current_predictions[score], row, 1, show_legend=False
    )
    fig.update_xaxes(title_text=f"{score} (score do modelo)", row=row, col=1)
    fig.update_yaxes(title_text="densidade", row=row, col=1)

    model = report.model
    _add_paired_bars(
        fig,
        ["chuva prevista"],
        [model.reference_positive_rate],
        [model.current_positive_rate],
        ".1%",
        ".1%",
        row,
        2,
    )
    fig.update_yaxes(title_text="% dos registros", tickformat=".0%", range=[0, 1.1], row=row, col=2)


def render_drift_report(
    report: PipelineReport,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_predictions: pd.DataFrame | None = None,
    current_predictions: pd.DataFrame | None = None,
) -> str:
    """Render the full interactive HTML page (Plotly embedded, works offline)."""
    figure_html = build_drift_figure(
        report, reference, current, reference_predictions, current_predictions
    ).to_html(full_html=False, include_plotlyjs=True)
    decision = report.decision
    model_text = (
        f" · modelo <code>{escape(report.model.model_version)}</code>" if report.model else ""
    )
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relatório de Data Drift</title>
<style>
body {{ margin: 0; background: {COLOR_SURFACE}; color: #0b0b0b;
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
header {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px 0; }}
h1 {{ font-size: 1.5rem; margin: 0 0 4px; }}
.meta {{ color: #52514e; font-size: .9rem; }}
.chart {{ max-width: 1200px; margin: 0 auto; padding: 0 8px; }}
</style>
</head>
<body>
<header>
<h1>Relatório interativo de Data Drift</h1>
<div class="meta">Execução <code>{escape(report.run_id)}</code> · {escape(report.timestamp)} (UTC)
 · alpha = {report.alpha} · {report.summary.drifted_features}/{report.summary.total_features}
 variáveis com drift{model_text} · ação: <code>{escape(decision.action)}</code></div>
</header>
<div class="chart">{figure_html}</div>
</body>
</html>
"""


def write_drift_report(
    report: PipelineReport,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    path: Path,
    reference_predictions: pd.DataFrame | None = None,
    current_predictions: pd.DataFrame | None = None,
) -> Path:
    """Write the interactive Plotly report."""
    html = render_drift_report(
        report, reference, current, reference_predictions, current_predictions
    )
    return _write_text(html, path)


# --------------------------------------------------------------------------- #
# Entry point: build the HTML reports from persisted artefacts only
# --------------------------------------------------------------------------- #
def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if path.is_file():
        return pd.read_csv(path)
    logger.warning("%s not found; the related charts will be omitted.", path.name)
    return None


def generate_html_reports(
    summary_path: Path = config.DATA_DIR / config.DRIFT_SUMMARY_FILE,
    data_dir: Path = config.DATA_DIR,
    reports_dir: Path = config.REPORTS_DIR,
) -> dict[str, Path]:
    """Read the audit log and data files, then write both HTML reports."""
    report = load_summary(summary_path)
    reference = load_dataset(data_dir / config.REFERENCE_DATA_FILE)
    current = load_dataset(data_dir / config.CURRENT_DATA_FILE)
    reference_predictions = _read_optional_csv(data_dir / config.REFERENCE_PREDICTIONS_FILE)
    current_predictions = _read_optional_csv(data_dir / config.CURRENT_PREDICTIONS_FILE)
    return {
        "exec_summary": write_exec_summary(report, reports_dir / config.EXEC_SUMMARY_FILE),
        "drift_report": write_drift_report(
            report,
            reference,
            current,
            reports_dir / config.DRIFT_REPORT_FILE,
            reference_predictions,
            current_predictions,
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    generate_html_reports()
