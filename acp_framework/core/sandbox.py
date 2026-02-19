from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_RUNNER_CODE = """
import json
import pathlib
import sys

_ALLOWED_MODULES = {"re", "math", "statistics", "string"}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root not in _ALLOWED_MODULES:
        raise ImportError(f"module not allowed: {name}")
    return __import__(name, globals, locals, fromlist, level)


def _safe_builtins():
    allowed = {
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "len": len,
        "range": range,
        "any": any,
        "all": all,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "enumerate": enumerate,
        "set": set,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "zip": zip,
        "sorted": sorted,
        "print": lambda *args, **kwargs: None,
        "__import__": _safe_import,
    }
    return allowed


def _load_rule(rule_path: pathlib.Path):
    code = rule_path.read_text(encoding="utf-8")
    namespace = {}
    exec(code, {"__builtins__": _safe_builtins()}, namespace)
    check = namespace.get("check")
    if not callable(check):
        raise ValueError("Rule must define callable check(x)")
    return check


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: runner.py <rule.py> <payload.json>")

    rule_path = pathlib.Path(sys.argv[1])
    payload_path = pathlib.Path(sys.argv[2])

    values = json.loads(payload_path.read_text(encoding="utf-8"))
    check = _load_rule(rule_path)

    mask = []
    for value in values:
        try:
            mask.append(bool(check(value)))
        except Exception:
            mask.append(False)

    sys.stdout.write(json.dumps({"mask": mask}, ensure_ascii=False))


if __name__ == "__main__":
    main()
""".strip()


class SandboxExecutionError(RuntimeError):
    """Raised when code execution in sandbox fails."""


@dataclass(slots=True)
class SandboxConfig:
    mode: str = "dev_local"
    firejail_bin: str = "firejail"
    timeout_s: int = 5
    python_bin: str = sys.executable


class SandboxRunner:
    """Execute untrusted rule code and return a boolean hit mask."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self._mode = self.config.mode.strip().lower()

        if self._mode not in {"dev_local", "strict_firejail"}:
            raise ValueError(f"Unsupported sandbox mode: {self.config.mode}")

        if self._mode == "strict_firejail" and shutil.which(self.config.firejail_bin) is None:
            raise RuntimeError(
                "strict_firejail mode enabled but firejail binary was not found. "
                "Install firejail or use mode=dev_local."
            )

    def execute_rule(self, code: str, values: Sequence[Any]) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="acp_sandbox_") as tmp:
            tmp_path = Path(tmp)
            rule_path = tmp_path / "rule.py"
            payload_path = tmp_path / "payload.json"
            runner_path = tmp_path / "runner.py"

            rule_path.write_text(code, encoding="utf-8")
            payload_path.write_text(json.dumps(list(values), ensure_ascii=False), encoding="utf-8")
            runner_path.write_text(_RUNNER_CODE, encoding="utf-8")

            cmd = self._build_command(tmp_path, runner_path, rule_path, payload_path)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
                check=False,
            )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            message = stderr or stdout or "unknown sandbox failure"
            raise SandboxExecutionError(message)

        payload = _extract_json_payload(proc.stdout or "")
        mask_raw = payload.get("mask")
        if not isinstance(mask_raw, list):
            raise SandboxExecutionError("Sandbox returned invalid mask payload")

        mask = np.asarray(mask_raw, dtype=bool)
        return mask

    def _build_command(self, tmp_path: Path, runner_path: Path, rule_path: Path, payload_path: Path) -> list[str]:
        base = [self.config.python_bin, str(runner_path), str(rule_path), str(payload_path)]
        if self._mode == "dev_local":
            return base

        return [
            self.config.firejail_bin,
            "--quiet",
            "--net=none",
            "--read-only=/",
            f"--private={tmp_path}",
            *base,
        ]


def _extract_json_payload(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise SandboxExecutionError("Sandbox produced empty stdout")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    matches = re.findall(r"\{[\s\S]*\}", text)
    for candidate in reversed(matches):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise SandboxExecutionError("Failed to parse JSON result from sandbox stdout")
