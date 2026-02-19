from __future__ import annotations

from pathlib import Path

from core.workspace import create_run_workspace, stage_user_input


def test_workspace_creation_and_stage_input_directory(tmp_path: Path) -> None:
    source = tmp_path / "demo"
    source.mkdir()
    (source / "train.csv").write_text("label,data\n1,a\n", encoding="utf-8")
    (source / "description.md").write_text("demo task", encoding="utf-8")
    runtime = create_run_workspace(tmp_path / "runs")
    staged = stage_user_input(source, runtime)

    assert runtime.workspace_dir.exists()
    assert staged.exists()
    assert (staged / "train.csv").exists()
    assert (staged / "description.md").exists()
