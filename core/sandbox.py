from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agents.coder_agent import RuleCandidate


_ALLOWED_IMPORT_PREFIXES = (
    "math",
    "re",
    "numpy",
    "tools",
)

_BLOCKED_CALLS = {
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "__import__",
}

_BLOCKED_ATTR_BASES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
}

_WRAPPER_SCRIPT = textwrap.dedent(
    """
    import importlib.util
    import json
    import sys
    from pathlib import Path

    def load_module(module_path: Path):
        spec = importlib.util.spec_from_file_location("sandbox_user_code", module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module from {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def main():
        code_path = Path(sys.argv[1])
        payload_path = Path(sys.argv[2])

        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        try:
            module = load_module(code_path)
            if not hasattr(module, "main") or not callable(module.main):
                raise ValueError("Sandboxed script must define callable main(payload)")
            result = module.main(payload)
            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))

    if __name__ == "__main__":
        main()
    """
).strip()

_RULE_VALIDATOR_SCRIPT = textwrap.dedent(
    """
    import contextlib
    import io

    ALLOWED_IMPORT_PREFIXES = ("math", "re", "numpy", "tools")

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        for prefix in ALLOWED_IMPORT_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Forbidden import: {name}")

    SAFE_BUILTINS = {
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "any": any,
        "all": all,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "__import__": restricted_import,
    }

    def main(payload):
        code = payload.get("candidate_code", "")
        values = payload.get("values", [])

        namespace = {}
        safe_globals = {"__builtins__": SAFE_BUILTINS}

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code, safe_globals, namespace)
            check = namespace.get("check") or safe_globals.get("check")
            if check is None or not callable(check):
                raise ValueError("Candidate code must define callable check(x)")

            mask = []
            for item in values:
                try:
                    mask.append(bool(check(item)))
                except Exception:
                    mask.append(False)

        return {"mask": mask}
    """
).strip()


@dataclass(slots=True)
class SandboxExecutionResult:
    ok: bool
    payload: Any
    error: str | None
    runtime_ms: int
    returncode: int
    stdout: str
    stderr: str


@dataclass(slots=True)
class RuleEvalResult:
    candidate: RuleCandidate
    valid: bool
    mask: np.ndarray
    error: str | None
    runtime_ms: int


def _extract_json_payload(stdout: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue

    start = stdout.find("{")
    end = stdout.rfind("}")
    if start >= 0 and end > start:
        chunk = stdout[start : end + 1]
        try:
            payload = json.loads(chunk)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            return None
    return None


def validate_candidate_safety(code: str) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    has_check = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            has_check = True

        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_import_allowed(alias.name):
                    return False, f"Forbidden import: {alias.name}"

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not _is_import_allowed(module):
                return False, f"Forbidden import-from: {module}"

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALLS:
                return False, f"Forbidden call: {node.func.id}"

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name) and base.id in _BLOCKED_ATTR_BASES:
                return False, f"Forbidden call: {base.id}.{node.func.attr}"

    if not has_check:
        return False, "Candidate must define function check(x)"

    return True, None


def _is_import_allowed(module: str) -> bool:
    if not module:
        return False

    for prefix in _ALLOWED_IMPORT_PREFIXES:
        if module == prefix or module.startswith(prefix + "."):
            return True
    return False


class LocalRestrictedRunner:
    def __init__(self, tmp_dir: str | Path, timeout_s: int = 5, project_root: str | Path | None = None) -> None:
        self.tmp_dir = Path(tmp_dir).resolve()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()

    def execute_code(self, code: str, payload: Any | None = None, timeout_s: int | None = None) -> SandboxExecutionResult:
        start = time.perf_counter()
        timeout = timeout_s if timeout_s is not None else self.timeout_s

        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as run_dir:
            run_root = Path(run_dir)
            code_path = run_root / "user_code.py"
            payload_path = run_root / "payload.json"
            launcher_path = run_root / "launcher.py"

            code_path.write_text(code, encoding="utf-8")
            payload_path.write_text(json.dumps(payload if payload is not None else {}, ensure_ascii=False), encoding="utf-8")
            launcher_path.write_text(_WRAPPER_SCRIPT + "\n", encoding="utf-8")

            cmd = self._build_command(run_root, launcher_path, code_path, payload_path)

            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root) + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            try:
                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                    check=False,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                runtime_ms = int((time.perf_counter() - start) * 1000)
                return SandboxExecutionResult(
                    ok=False,
                    payload=None,
                    error=f"Timeout after {timeout}s",
                    runtime_ms=runtime_ms,
                    returncode=124,
                    stdout="",
                    stderr="",
                )

            runtime_ms = int((time.perf_counter() - start) * 1000)
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""

            if completed.returncode != 0:
                return SandboxExecutionResult(
                    ok=False,
                    payload=None,
                    error=f"Executor failed ({completed.returncode}): {stderr.strip()[:500]}",
                    runtime_ms=runtime_ms,
                    returncode=completed.returncode,
                    stdout=stdout,
                    stderr=stderr,
                )

            payload_json = _extract_json_payload(stdout)
            if payload_json is None:
                return SandboxExecutionResult(
                    ok=False,
                    payload=None,
                    error="No JSON result found on sandbox stdout",
                    runtime_ms=runtime_ms,
                    returncode=completed.returncode,
                    stdout=stdout,
                    stderr=stderr,
                )

            if not bool(payload_json.get("ok", False)):
                return SandboxExecutionResult(
                    ok=False,
                    payload=None,
                    error=str(payload_json.get("error", "Unknown sandbox error")),
                    runtime_ms=runtime_ms,
                    returncode=completed.returncode,
                    stdout=stdout,
                    stderr=stderr,
                )

            return SandboxExecutionResult(
                ok=True,
                payload=payload_json.get("result"),
                error=None,
                runtime_ms=runtime_ms,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
            )

    def validate_rules(self, rule_candidates: Sequence[RuleCandidate], data_values: Sequence[object]) -> list[RuleEvalResult]:
        values = list(data_values)
        results: list[RuleEvalResult] = []

        for candidate in rule_candidates:
            safe, reason = validate_candidate_safety(candidate.code)
            if not safe:
                results.append(
                    RuleEvalResult(
                        candidate=candidate,
                        valid=False,
                        mask=np.zeros(len(values), dtype=bool),
                        error=reason,
                        runtime_ms=0,
                    )
                )
                continue

            exec_result = execute_code(
                self,
                _RULE_VALIDATOR_SCRIPT,
                {"candidate_code": candidate.code, "values": values},
                timeout_s=self.timeout_s,
            )

            if not exec_result.ok:
                results.append(
                    RuleEvalResult(
                        candidate=candidate,
                        valid=False,
                        mask=np.zeros(len(values), dtype=bool),
                        error=exec_result.error,
                        runtime_ms=exec_result.runtime_ms,
                    )
                )
                continue

            payload = exec_result.payload
            if not isinstance(payload, dict):
                results.append(
                    RuleEvalResult(
                        candidate=candidate,
                        valid=False,
                        mask=np.zeros(len(values), dtype=bool),
                        error="Sandbox payload is not a dictionary",
                        runtime_ms=exec_result.runtime_ms,
                    )
                )
                continue

            mask = np.array(payload.get("mask", []), dtype=bool)
            if mask.size != len(values):
                results.append(
                    RuleEvalResult(
                        candidate=candidate,
                        valid=False,
                        mask=np.zeros(len(values), dtype=bool),
                        error="Invalid mask length returned by candidate",
                        runtime_ms=exec_result.runtime_ms,
                    )
                )
                continue

            results.append(
                RuleEvalResult(
                    candidate=candidate,
                    valid=True,
                    mask=mask,
                    error=None,
                    runtime_ms=exec_result.runtime_ms,
                )
            )

        return results

    def _build_command(
        self,
        run_root: Path,
        launcher_path: Path,
        code_path: Path,
        payload_path: Path,
    ) -> list[str]:
        _ = run_root
        return [
            sys.executable,
            str(launcher_path),
            str(code_path),
            str(payload_path),
        ]


class FirejailSandboxRunner(LocalRestrictedRunner):
    def __init__(
        self,
        tmp_dir: str | Path,
        timeout_s: int = 5,
        project_root: str | Path | None = None,
        firejail_bin: str = "firejail",
        rlimit_as_mb: int = 4096,
    ) -> None:
        super().__init__(tmp_dir=tmp_dir, timeout_s=timeout_s, project_root=project_root)
        self.firejail_bin = firejail_bin
        self.rlimit_as_mb = rlimit_as_mb

    def _build_command(
        self,
        run_root: Path,
        launcher_path: Path,
        code_path: Path,
        payload_path: Path,
    ) -> list[str]:
        base = super()._build_command(run_root, launcher_path, code_path, payload_path)
        args = [
            self.firejail_bin,
            "--quiet",
            "--noprofile",
            "--net=none",
            "--read-only=/",
            f"--private={run_root}",
            "--private-dev",
            "--nosound",
        ]
        if self.rlimit_as_mb > 0:
            args.append(f"--rlimit-as={self.rlimit_as_mb * 1024 * 1024}")
        return [*args, *base]


def build_sandbox_runner(
    tmp_dir: str | Path,
    timeout_s: int,
    project_root: str | Path,
    mode: str = "strict_firejail",
    firejail_bin: str = "firejail",
    rlimit_as_mb: int = 4096,
) -> LocalRestrictedRunner:
    normalized_mode = mode.strip().lower()

    if normalized_mode == "strict_firejail":
        if shutil.which(firejail_bin) is None:
            raise RuntimeError(
                f"Fatal: sandbox.mode=strict_firejail but binary '{firejail_bin}' was not found in PATH. "
                "Install firejail or set sandbox.mode=dev_local for local development."
            )
        return FirejailSandboxRunner(
            tmp_dir=tmp_dir,
            timeout_s=timeout_s,
            project_root=project_root,
            firejail_bin=firejail_bin,
            rlimit_as_mb=rlimit_as_mb,
        )

    if normalized_mode == "dev_local":
        return LocalRestrictedRunner(tmp_dir=tmp_dir, timeout_s=timeout_s, project_root=project_root)

    raise ValueError(f"Unknown sandbox mode: {mode}")


def execute_code(
    runner: LocalRestrictedRunner,
    code: str,
    payload: Any | None = None,
    timeout_s: int | None = None,
) -> SandboxExecutionResult:
    return runner.execute_code(code=code, payload=payload, timeout_s=timeout_s)
