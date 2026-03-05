"""Unified Discovery — Lightweight, deterministic data structure probing for all layers."""

from typing import Any, Dict, Optional

from core.sandbox import CodeSandbox


def run_discovery(
    sandbox: CodeSandbox,
    layer: str,
    task_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute lightweight discovery and return result.

    Generates hardcoded Python code that prints:
    - DataFrame structure: shape, dtypes, head(3) (raw repr to catch nested structures)
    - Per-column: dtype, missing count, unique count, 3 sample values (repr to 200 chars)
    - Nested detection: flags columns where sample values are list/dict/ndarray
    - File inventory: extension counts, directory structure
    - Media samples: image/audio properties (if DATA_DIR is set)
    - Target column info (for L2/L3)

    Args:
        sandbox: CodeSandbox instance with df and optionally DATA_DIR loaded.
        layer: Layer identifier ("L0", "L1", "L2", "L3").
        task_config: Task configuration dict (used for target info in L2/L3).

    Returns:
        Sandbox execution result dict with stdout, success, _code keys.
    """
    target = ""
    if task_config and layer in ("L2", "L3"):
        target = task_config.get("target", "") or ""

    target_block = ""
    if target:
        target_block = f'''
# Target column details
target_col = "{target}"
if target_col in df.columns:
    t = df[target_col]
    print(f"\\n=== Target Column: {{target_col}} ===")
    print(f"  dtype: {{t.dtype}}")
    print(f"  missing: {{t.isna().sum()}} / {{len(df)}}")
    print(f"  unique: {{t.nunique()}}")
    print(f"  samples: {{[repr(v) for v in t.dropna().head(5).tolist()]}}")
    if t.dtype == 'object' or t.nunique() < 30:
        print(f"  value_counts (top 10): {{dict(t.value_counts().head(10))}}")
'''

    code = f'''import pandas as pd
import numpy as np

# === DataFrame Discovery ===
if 'df' in dir() and df is not None:
    print("=== DataFrame Structure ===")
    print(f"Shape: {{df.shape[0]}} rows, {{df.shape[1]}} columns")
    print(f"Dtypes: {{dict(df.dtypes.value_counts())}}")
    print()
    print("=== Head (3 rows, raw repr) ===")
    print(repr(df.head(3)))
    print()

    # Per-column analysis with nested structure detection
    print("=== Column Discovery ===")
    nested_columns = []
    for col in df.columns:
        c = df[col]
        missing = int(c.isna().sum())
        unique = int(c.nunique())
        dtype = str(c.dtype)

        # Get sample values as repr (catches lists, dicts, arrays)
        samples_raw = c.dropna().head(3).tolist()
        samples_repr = [repr(v) for v in samples_raw]

        # Detect nested structures
        is_nested = False
        for v in samples_raw:
            if isinstance(v, (list, dict)):
                is_nested = True
                break
            if isinstance(v, np.ndarray):
                is_nested = True
                break
            if isinstance(v, str) and len(v) > 1 and v[0] in ('[', '{{'):
                is_nested = True
                break

        flag = " [NESTED]" if is_nested else ""
        if is_nested:
            nested_columns.append(col)

        print(f"Column: {{col}}{{flag}}")
        print(f"  dtype={{dtype}}, missing={{missing}} ({{missing/len(df)*100:.1f}}%), unique={{unique}}")
        print(f"  samples: {{samples_repr}}")

    if nested_columns:
        print(f"\\n** Nested/complex columns detected: {{nested_columns}}")
        print("   These columns contain lists, dicts, or arrays — not flat scalars.")

    {target_block}
else:
    print("=== No DataFrame loaded ===")

# === File Inventory ===
if 'DATA_DIR' in dir():
    from pathlib import Path
    data_dir = Path(DATA_DIR)
    print("\\n=== File Inventory ===")
    ext_counts = {{}}
    total_size = 0
    for f in data_dir.rglob("*"):
        if f.is_file() and not f.name.startswith(".") and f.name != "description.md":
            ext = f.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            total_size += f.stat().st_size
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"  {{ext}}: {{count}} files")
    print(f"Total: {{sum(ext_counts.values())}} files, {{total_size / 1e6:.1f}} MB")

    # Media samples (if any)
    image_exts = {{".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}}
    audio_exts = {{".wav", ".mp3", ".flac", ".ogg", ".m4a"}}
    image_files = sorted([f for f in data_dir.rglob("*") if f.suffix.lower() in image_exts])
    audio_files = sorted([f for f in data_dir.rglob("*") if f.suffix.lower() in audio_exts])

    if image_files:
        print(f"\\n=== Image Samples ({{len(image_files)}} total) ===")
        if 'batch_analyze_images' in dir():
            step = max(1, len(image_files) // 5)
            results = batch_analyze_images(image_files[::step][:5])
            for r in results:
                if 'width' in r:
                    print(f"  {{Path(r['path']).name}}: {{r['width']}}x{{r['height']}} {{r.get('mode', '?')}}")
                elif 'error' in r:
                    print(f"  {{Path(r['path']).name}}: ERROR {{r['error']}}")
        else:
            for fp in image_files[:3]:
                print(f"  {{fp.name}} ({{fp.stat().st_size}} bytes)")

    if audio_files:
        print(f"\\n=== Audio Samples ({{len(audio_files)}} total) ===")
        if 'batch_analyze_audio' in dir():
            step = max(1, len(audio_files) // 5)
            results = batch_analyze_audio(audio_files[::step][:5])
            for r in results:
                if 'duration_sec' in r:
                    print(f"  {{Path(r['path']).name}}: {{r['sample_rate']}}Hz, {{r['channels']}}ch, {{r['duration_sec']}}s")
                elif 'error' in r:
                    print(f"  {{Path(r['path']).name}}: ERROR {{r['error']}}")
        else:
            for fp in audio_files[:3]:
                print(f"  {{fp.name}} ({{fp.stat().st_size}} bytes)")
'''

    result = sandbox.execute(code, f"Discovery ({layer})")
    result["_code"] = code
    return result
