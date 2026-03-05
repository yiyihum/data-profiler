"""Code Sandbox - Isolated execution environment for LLM-generated code."""

import sys
import traceback
from io import StringIO
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np


class CodeSandbox:
    """
    Executes LLM-generated Python code in an isolated environment.

    Features:
    - Prevents modification of original data files
    - Captures stdout, stderr, and return values
    - Supports retry on failure (max N=3)
    - Returns structured execution results
    - Multimodal support: image/audio analysis helpers with GPU access
    """

    MAX_RETRIES = 3

    def __init__(self, output_dir: Path, data_dir: Optional[Path] = None):
        """
        Initialize sandbox with output directory for artifacts.

        Args:
            output_dir: Directory where generated plots/files will be saved.
            data_dir: Optional path to the data directory (for multimodal access).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_dir) if data_dir else None

        # Shared namespace for code execution (persists across calls)
        self._namespace: Dict[str, Any] = {
            "pd": pd,
            "np": np,
            "__builtins__": __builtins__,
        }

        # Import common libraries into namespace
        self._setup_namespace()

        # Inject data_dir if provided
        if self.data_dir:
            self._namespace["DATA_DIR"] = str(self.data_dir)

    def _setup_namespace(self) -> None:
        """Setup common imports in the execution namespace."""
        setup_code = """
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')
"""
        try:
            exec(setup_code, self._namespace)
        except ImportError as e:
            print(f"Warning: Some imports failed: {e}")

    def setup_multimodal(self, gpu_id: int = 2) -> None:
        """
        Inject multimodal libraries into the sandbox namespace.

        Sets CUDA_VISIBLE_DEVICES and imports torch, PIL, pathlib, torchaudio.
        Called from main.py when non-tabular data is detected.
        """
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        setup_code = f"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

import torch
from PIL import Image
from pathlib import Path
import json
"""
        try:
            exec(setup_code, self._namespace)
            print(f"[SANDBOX] Multimodal setup complete (GPU {gpu_id})")
        except ImportError as e:
            print(f"[SANDBOX] Warning: Some multimodal imports failed: {e}")

    def inject_helpers(self) -> None:
        """Inject multimodal helper functions into the sandbox namespace."""
        helper_code = '''
def analyze_image(path, max_size=512):
    """Load image, return basic info dict + PIL Image (resized)."""
    from PIL import Image as _PILImage
    img = _PILImage.open(str(path))
    info = {"width": img.width, "height": img.height, "mode": img.mode, "format": img.format}
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))
    return info, img

def analyze_audio(path):
    """Load audio, return basic info dict + waveform tensor."""
    import torchaudio as _ta
    waveform, sr = _ta.load(str(path))
    duration = waveform.shape[1] / sr
    return {"sample_rate": sr, "channels": waveform.shape[0],
            "duration_sec": round(duration, 2), "samples": waveform.shape[1]}, waveform

def list_data_files(pattern="**/*", extensions=None):
    """List files in DATA_DIR matching pattern and optional extension filter."""
    from pathlib import Path as _P
    data_dir = _P(DATA_DIR)
    files = sorted(data_dir.glob(pattern))
    files = [f for f in files if f.is_file()]
    if extensions:
        ext_set = {e.lower() for e in extensions}
        files = [f for f in files if f.suffix.lower() in ext_set]
    return files

def batch_analyze_images(paths, max_n=20):
    """Analyze multiple images, return list of info dicts."""
    results = []
    for p in list(paths)[:max_n]:
        try:
            info, _ = analyze_image(p)
            info["path"] = str(p)
            results.append(info)
        except Exception as e:
            results.append({"error": str(e), "path": str(p)})
    return results

def batch_analyze_audio(paths, max_n=20):
    """Analyze multiple audio files, return list of info dicts."""
    results = []
    for p in list(paths)[:max_n]:
        try:
            info, _ = analyze_audio(p)
            info["path"] = str(p)
            results.append(info)
        except Exception as e:
            results.append({"error": str(e), "path": str(p)})
    return results

def run_hf_pipeline(task, model=None, inputs=None, **kwargs):
    """Run a HuggingFace transformers pipeline. GPU used if available."""
    from transformers import pipeline as _hf_pipeline
    import torch as _torch
    device = 0 if _torch.cuda.is_available() else -1
    pipe = _hf_pipeline(task, model=model, device=device, **kwargs)
    return pipe(inputs)
'''
        try:
            exec(helper_code, self._namespace)
            print("[SANDBOX] Multimodal helpers injected")
        except Exception as e:
            print(f"[SANDBOX] Warning: Failed to inject helpers: {e}")

    def load_dataframe(self, data_path: Path, var_name: str = "df", max_rows: int = 0) -> Tuple[bool, str]:
        """
        Load a dataset into the sandbox namespace.

        Args:
            data_path: Path to CSV or Parquet file.
            var_name: Variable name to use in the namespace.
            max_rows: If >0, cap the loaded rows (random sample for large files).

        Returns:
            Tuple of (success, message).
        """
        data_path = Path(data_path)

        if not data_path.exists():
            return False, f"File not found: {data_path}"

        try:
            ext = data_path.suffix.lower()
            if ext == ".parquet":
                df = pd.read_parquet(data_path)
            elif ext in (".csv", ".txt", ".tsv"):
                sep = "\t" if ext == ".tsv" else ","
                df = pd.read_csv(data_path, nrows=max_rows if max_rows > 0 else None, sep=sep)
            elif ext in (".json", ".jsonl", ".ndjson"):
                try:
                    df = pd.read_json(data_path, lines=(ext in (".jsonl", ".ndjson")))
                except ValueError:
                    # Fallback: try lines=True for regular .json that is actually JSONL
                    df = pd.read_json(data_path, lines=True)
                if max_rows > 0 and len(df) > max_rows:
                    df = df.head(max_rows)
            elif ext == ".feather":
                df = pd.read_feather(data_path)
            else:
                return False, f"Unsupported file format: {data_path.suffix}"

            sampled = ""
            if max_rows > 0 and len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
                sampled = f" (sampled from larger dataset)"

            # Store original and working copy
            self._namespace[f"{var_name}_original"] = df.copy()
            self._namespace[var_name] = df.copy()
            self._namespace["OUTPUT_DIR"] = str(self.output_dir)

            return True, f"Loaded {len(df)} rows, {len(df.columns)} columns into '{var_name}'{sampled}"

        except Exception as e:
            return False, f"Failed to load data: {str(e)}"

    def execute(self, code: str, description: str = "") -> Dict[str, Any]:
        """
        Execute Python code in the sandbox.

        Args:
            code: Python code to execute.
            description: Human-readable description of what the code does.

        Returns:
            Dictionary with execution results:
            - success: bool
            - stdout: captured print output
            - result: last expression result (if any)
            - error: error message (if failed)
            - artifacts: list of generated file paths
        """
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_stdout = StringIO()

        result = {
            "success": False,
            "stdout": "",
            "result": None,
            "error": None,
            "artifacts": [],
            "description": description
        }

        try:
            # Execute code
            exec(code, self._namespace)

            # Check for generated plots
            if "plt" in self._namespace:
                plt = self._namespace["plt"]
                if plt.get_fignums():
                    # Save any open figures
                    for i, fig_num in enumerate(plt.get_fignums()):
                        fig = plt.figure(fig_num)
                        artifact_path = self.output_dir / f"plot_{len(result['artifacts'])+1}.png"
                        fig.savefig(artifact_path, dpi=150, bbox_inches='tight')
                        result["artifacts"].append(str(artifact_path))
                    plt.close('all')

            result["success"] = True
            result["stdout"] = captured_stdout.getvalue()

            # Try to get result of last expression
            if "_result" in self._namespace:
                result["result"] = self._namespace.pop("_result")

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result["stdout"] = captured_stdout.getvalue()

        finally:
            sys.stdout = old_stdout

        return result

    def execute_with_retry(
        self,
        code: str,
        description: str = "",
        fix_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Execute code with automatic retry on failure.

        Args:
            code: Python code to execute.
            description: Description of the code.
            fix_callback: Optional callback(code, error) -> fixed_code for LLM-based fixes.

        Returns:
            Execution result dictionary.
        """
        current_code = code
        last_result = None

        for attempt in range(self.MAX_RETRIES):
            result = self.execute(current_code, description)
            last_result = result

            if result["success"]:
                return result

            # If we have a fix callback, try to fix the code
            if fix_callback and attempt < self.MAX_RETRIES - 1:
                try:
                    fixed_code = fix_callback(current_code, result["error"])
                    if fixed_code and fixed_code != current_code:
                        current_code = fixed_code
                        continue
                except Exception:
                    pass

            # No fix available or fix didn't help
            break

        # Mark as failed after retries
        last_result["retries_exhausted"] = True
        return last_result

    def get_variable(self, name: str) -> Any:
        """Get a variable from the sandbox namespace."""
        return self._namespace.get(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the sandbox namespace."""
        self._namespace[name] = value

    def get_dataframe(self, name: str = "df") -> Optional[pd.DataFrame]:
        """Get a DataFrame from the namespace."""
        return self._namespace.get(name)

    def reset_dataframe(self, name: str = "df") -> None:
        """Reset DataFrame to original loaded state."""
        original_name = f"{name}_original"
        if original_name in self._namespace:
            self._namespace[name] = self._namespace[original_name].copy()
