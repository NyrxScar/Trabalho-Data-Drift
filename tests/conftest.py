"""Shared pytest fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from src.decision import PipelineReport
from src.generate_data import generate_datasets
from src.main import run_pipeline


@dataclass(frozen=True)
class PipelineRun:
    """Directories and report of a full pipeline run in a temporary folder."""

    data_dir: Path
    reports_dir: Path
    models_dir: Path
    report: PipelineReport


@pytest.fixture(scope="session")
def datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference and production datasets generated with the default seed."""
    return generate_datasets()


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory) -> PipelineRun:
    """One end-to-end execution (data, training, inference, monitoring, reports)."""
    root = tmp_path_factory.mktemp("pipeline")
    dirs = {name: root / name for name in ("data", "reports", "models")}
    report = run_pipeline(
        seed=42,
        alpha=0.05,
        data_dir=dirs["data"],
        reports_dir=dirs["reports"],
        models_dir=dirs["models"],
    )
    return PipelineRun(dirs["data"], dirs["reports"], dirs["models"], report)


@pytest.fixture(scope="session")
def pipeline_report(pipeline_run: PipelineRun) -> PipelineReport:
    """Consolidated report computed from the default datasets."""
    return pipeline_run.report
