import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def output(self) -> str:
        if self.success:
            return self.stdout
        return f"STDOUT:\n{self.stdout}\nSTDERR:\n{self.stderr}"


class CodeExecutor:
    def __init__(self, config: Config):
        self.config = config
        self.timeout = config.code_timeout
        self.firejail = config.firejail_enabled

    def execute(self, code: str, data_path: str = None) -> ExecutionResult:
        """Execute Python code in a subprocess (optionally sandboxed with firejail)."""
        if data_path is None:
            data_path = self.config.data_path

        # Prepend data_path as a variable
        full_code = f'DATA_PATH = "{data_path}"\n' + code

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            tmp_path = f.name

        try:
            if self.firejail:
                cmd = [
                    "firejail", "--noprofile", "--net=none",
                    f"--whitelist={os.path.dirname(data_path)}",
                    f"--read-only={os.path.dirname(data_path)}",
                    f"--whitelist={os.path.dirname(tmp_path)}",
                    "python", tmp_path,
                ]
            else:
                cmd = ["python", tmp_path]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.path.dirname(data_path),
            )
            return ExecutionResult(
                stdout=result.stdout[:5000],
                stderr=result.stderr[:2000],
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {self.timeout} seconds",
                exit_code=-1,
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
            )
        finally:
            os.unlink(tmp_path)
