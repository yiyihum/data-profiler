from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agents.coder_agent import RuleCandidate
from core.sandbox import LocalRestrictedRunner, build_sandbox_runner, execute_code


def test_sandbox_rejects_forbidden_import(tmp_path: Path) -> None:
    runner = LocalRestrictedRunner(tmp_dir=tmp_path, timeout_s=2, project_root=Path.cwd())
    bad = RuleCandidate(description="bad", code="import os\n\ndef check(x):\n    return True\n")

    results = runner.validate_rules([bad], ["a", "b"])

    assert len(results) == 1
    assert not results[0].valid
    assert "Forbidden import" in (results[0].error or "")


def test_sandbox_executes_valid_rule(tmp_path: Path) -> None:
    runner = LocalRestrictedRunner(tmp_dir=tmp_path, timeout_s=2, project_root=Path.cwd())
    good = RuleCandidate(
        description="len",
        code="def check(x):\n    text = '' if x is None else str(x)\n    return len(text) > 3\n",
    )

    results = runner.validate_rules([good], ["a", "abcd"])

    assert results[0].valid
    assert results[0].mask.tolist() == [False, True]


def test_execute_code_cleans_stdout_and_parses_json(tmp_path: Path) -> None:
    runner = LocalRestrictedRunner(tmp_dir=tmp_path, timeout_s=2, project_root=Path.cwd())
    code = (
        "def main(payload):\n"
        "    print('debug log from script')\n"
        "    return {'echo': payload.get('x', 0)}\n"
    )
    result = execute_code(runner, code, {"x": 7})
    assert result.ok
    assert isinstance(result.payload, dict)
    assert result.payload["echo"] == 7


def test_sandbox_timeout_for_infinite_loop(tmp_path: Path) -> None:
    runner = LocalRestrictedRunner(tmp_dir=tmp_path, timeout_s=1, project_root=Path.cwd())
    hanging = RuleCandidate(
        description="hang",
        code="def check(x):\n    while True:\n        pass\n",
    )
    results = runner.validate_rules([hanging], [1])
    assert not results[0].valid
    assert "Timeout" in (results[0].error or "")


def test_sandbox_rejects_subprocess_import(tmp_path: Path) -> None:
    runner = LocalRestrictedRunner(tmp_dir=tmp_path, timeout_s=2, project_root=Path.cwd())
    bad = RuleCandidate(
        description="subprocess",
        code="import subprocess\n\ndef check(x):\n    return True\n",
    )
    results = runner.validate_rules([bad], [1])
    assert not results[0].valid
    assert "Forbidden import" in (results[0].error or "")


def test_sandbox_rejects_socket_import(tmp_path: Path) -> None:
    runner = LocalRestrictedRunner(tmp_dir=tmp_path, timeout_s=2, project_root=Path.cwd())
    bad = RuleCandidate(
        description="network",
        code="import socket\n\ndef check(x):\n    return True\n",
    )
    results = runner.validate_rules([bad], [1])
    assert not results[0].valid
    assert "Forbidden import" in (results[0].error or "")


def test_sandbox_rejects_filesystem_write_call(tmp_path: Path) -> None:
    runner = LocalRestrictedRunner(tmp_dir=tmp_path, timeout_s=2, project_root=Path.cwd())
    bad = RuleCandidate(
        description="file-write",
        code="def check(x):\n    f = open('/tmp/pwned', 'w')\n    f.write('x')\n    return True\n",
    )
    results = runner.validate_rules([bad], [1])
    assert not results[0].valid
    assert "Forbidden call: open" in (results[0].error or "")


def test_sandbox_rejects_dynamic_import_call(tmp_path: Path) -> None:
    runner = LocalRestrictedRunner(tmp_dir=tmp_path, timeout_s=2, project_root=Path.cwd())
    bad = RuleCandidate(
        description="dynamic-import",
        code="def check(x):\n    m = __import__('os')\n    return bool(m)\n",
    )
    results = runner.validate_rules([bad], [1])
    assert not results[0].valid
    assert "Forbidden call: __import__" in (results[0].error or "")


def test_strict_mode_fails_if_firejail_missing(tmp_path: Path) -> None:
    with patch("core.sandbox.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="strict_firejail"):
            build_sandbox_runner(
                tmp_dir=tmp_path,
                timeout_s=2,
                project_root=Path.cwd(),
                mode="strict_firejail",
                firejail_bin="firejail",
            )
