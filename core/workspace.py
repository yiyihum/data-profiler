from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class RuntimeContext:
    run_id: str
    workspace_dir: Path
    input_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    tmp_dir: Path


def create_run_workspace(workspace_root: str | Path, run_id: str | None = None) -> RuntimeContext:
    root = Path(workspace_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if run_id:
        rid = run_id
    else:
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
        rid = base
        counter = 1
        while (root / rid).exists():
            rid = f"{base}_{counter:02d}"
            counter += 1

    workspace_dir = root / rid / "workspace"
    input_dir = workspace_dir / "input"
    artifacts_dir = workspace_dir / "artifacts"
    logs_dir = workspace_dir / "logs"
    tmp_dir = workspace_dir / "tmp"

    for path in (input_dir, artifacts_dir, logs_dir, tmp_dir):
        path.mkdir(parents=True, exist_ok=False)

    return RuntimeContext(
        run_id=rid,
        workspace_dir=workspace_dir,
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        tmp_dir=tmp_dir,
    )


def stage_user_input(raw_path: str | Path, runtime: RuntimeContext, dst_name: str = "user_data") -> Path:
    source = Path(raw_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    destination = runtime.input_dir / dst_name
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    if source.is_file():
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source, destination / source.name)
    else:
        shutil.copytree(source, destination, dirs_exist_ok=False)

    return destination
