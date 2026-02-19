from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.ingestion_agent import IngestionAgent
from core.sandbox import LocalRestrictedRunner


def test_ingestion_standardizes_columns(tmp_path: Path) -> None:
    fixture = Path("scripts/fixtures/tiny_train.csv").resolve()
    sandbox = LocalRestrictedRunner(tmp_dir=tmp_path / "tmp", timeout_s=3, project_root=Path.cwd())
    agent = IngestionAgent(sandbox_runner=sandbox)

    result = agent.run(fixture, tmp_path)

    assert result.parquet_path.exists()
    frame = pd.read_parquet(result.parquet_path)
    assert list(frame.columns) == ["data", "label"]
    assert frame.shape[0] == 6
    assert result.metadata["task_description"] == "analyze the dataset"
    assert result.metadata["task_description_source"] == "default"
    assert result.metadata["task_description_path"] is None


def test_ingestion_uses_description_file_when_present(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("label,comment\n1,you are bad\n0,hello\n", encoding="utf-8")
    (data_dir / "description.md").write_text("Detect insulting comments in social text.", encoding="utf-8")

    sandbox = LocalRestrictedRunner(tmp_dir=tmp_path / "tmp2", timeout_s=3, project_root=Path.cwd())
    agent = IngestionAgent(sandbox_runner=sandbox)

    result = agent.run(data_dir, tmp_path / "out")

    assert result.metadata["task_description_source"] == "file"
    assert "insulting comments" in result.metadata["task_description"]
    assert str(data_dir / "description.md") in str(result.metadata["task_description_path"])
