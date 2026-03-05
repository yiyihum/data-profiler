"""
Data Profiler MVP - Main Entry Point

A state-machine based data profiling pipeline that:
1. Analyzes raw data quality (L0)
2. Explores feature structure (L1)
3. Aligns features with prediction task (L2)
4. Generates modeling strategy (L3)
5. Produces MLE report and preprocessing pipeline

Input: a directory containing description.md + data files (CSV/Parquet).
The pipeline auto-reads description.md as the task description and discovers
data files, columns, and sample values automatically. Target column, task type,
and metric are inferred by the LLM from the description and data.
"""

import argparse
import json
import sys
from pathlib import Path

import subprocess
import pandas as pd

from core.agent_loop import AgentLoopConfig
from core.state import StateContext
from core.sandbox import CodeSandbox
from core.llm import LLMClient, LLMConfig
from core.modality import detect_modality, sample_files, get_file_basic_info, sniff_file_structure, DataModality
from layers import L0Cleaner, L1Explorer, L2Aligner, L3Strategist
from report.generator import (
    ReportGenerator,
    generate_layer_diagnostics,
    generate_preprocessing_script,
    generate_execution_trace,
)


# Maximum rows to load; larger datasets are sampled down
MAX_ROWS = 200_000


def _fast_csv_row_count(fpath: Path) -> int:
    """Count CSV rows using wc -l (fast, avoids reading into Python)."""
    try:
        result = subprocess.run(
            ["wc", "-l", str(fpath)],
            capture_output=True, text=True, timeout=30,
        )
        return max(int(result.stdout.strip().split()[0]) - 1, 0)
    except Exception:
        return -1  # unknown


_TASK_INFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": ["string", "null"]},
        "task_type": {"type": "string"},
        "metric": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["target", "task_type", "metric", "description"],
    "additionalProperties": False,
}


def discover_data_dir(data_dir: Path) -> dict:
    """
    Auto-discover contents of a data directory.

    Reads description.md, detects data modality (tabular/image/audio/mixed),
    and scans CSV/Parquet files for column names and sample values.

    Returns:
        Dictionary with 'description', 'files' (list of file inventories),
        'primary_data_path' (the main training file or None),
        and 'modality' (DataModality object).
    """
    data_dir = Path(data_dir)

    # Read description.md
    description = ""
    desc_path = data_dir / "description.md"
    if desc_path.exists():
        description = desc_path.read_text(encoding="utf-8")

    # Detect modality
    modality = detect_modality(data_dir)

    # Scan tabular files for column names / samples
    files_info = []
    primary_data_path = None

    for fpath in modality.tabular_files:
        try:
            if fpath.suffix.lower() in (".csv", ".tsv"):
                sep = "\t" if fpath.suffix.lower() == ".tsv" else ","
                df_sample = pd.read_csv(fpath, nrows=5, sep=sep)
                n_rows_full = _fast_csv_row_count(fpath)
            else:
                df_sample = pd.read_parquet(fpath).head(5)
                import pyarrow.parquet as pq
                n_rows_full = pq.read_metadata(fpath).num_rows

            columns = df_sample.columns.tolist()
            dtypes = {col: str(df_sample[col].dtype) for col in columns}
            samples = {}
            for col in columns:
                vals = df_sample[col].dropna().head(3).tolist()
                samples[col] = [str(v)[:80] for v in vals]

            info = {
                "filename": fpath.name,
                "rows": n_rows_full,
                "columns": columns,
                "dtypes": dtypes,
                "samples": samples,
            }
            files_info.append(info)

            # Heuristic: pick the largest CSV/Parquet as primary (prefer "train" in name)
            if primary_data_path is None:
                primary_data_path = fpath
            elif "train" in fpath.stem.lower():
                primary_data_path = fpath
            elif "train" not in primary_data_path.stem.lower() and n_rows_full > info.get("rows", 0):
                primary_data_path = fpath

        except Exception as e:
            files_info.append({
                "filename": fpath.name,
                "error": str(e),
            })

    # Scan JSON files — try loading as tabular, else record structure
    for fpath in modality.json_files:
        try:
            is_lines = fpath.suffix.lower() in (".jsonl", ".ndjson")
            try:
                df_sample = pd.read_json(fpath, lines=is_lines, nrows=5 if is_lines else None)
                if not is_lines:
                    df_sample = df_sample.head(5)
            except ValueError:
                df_sample = pd.read_json(fpath, lines=True, nrows=5)

            n_rows_full = fpath.stat().st_size  # approximate; real count from wc
            if is_lines or fpath.suffix.lower() == ".json":
                try:
                    wc = _fast_csv_row_count(fpath)
                    if wc > 0:
                        n_rows_full = wc
                except Exception:
                    pass

            columns = df_sample.columns.tolist()
            dtypes = {col: str(df_sample[col].dtype) for col in columns}
            samples = {}
            for col in columns:
                vals = df_sample[col].dropna().head(3).tolist()
                samples[col] = [repr(v)[:80] for v in vals]

            info = {
                "filename": fpath.name,
                "rows": n_rows_full,
                "columns": columns,
                "dtypes": dtypes,
                "samples": samples,
                "format": "json",
            }
            files_info.append(info)

            # Pick JSON as primary if no CSV/Parquet found, or if "train" in name
            if primary_data_path is None:
                primary_data_path = fpath
            elif "train" in fpath.stem.lower() and "train" not in primary_data_path.stem.lower():
                primary_data_path = fpath

        except Exception:
            # JSON can't be loaded as flat table — record structure only
            struct_info = sniff_file_structure(fpath)
            files_info.append({
                "filename": fpath.name,
                "format": "json_non_tabular",
                "structure": struct_info,
            })
            # Still consider as primary if nothing else and "train" in name
            if primary_data_path is None and "train" in fpath.stem.lower():
                primary_data_path = fpath

    return {
        "description": description,
        "files": files_info,
        "primary_data_path": primary_data_path,
        "modality": modality,
    }


def build_file_inventory_text(files_info: list, modality: DataModality = None) -> str:
    """Format file inventory as readable text for LLM context."""
    parts = []

    # Tabular files detail
    for fi in files_info:
        if "error" in fi:
            parts.append(f"File: {fi['filename']} — error reading: {fi['error']}")
            continue
        parts.append(f"File: {fi['filename']} ({fi['rows']} rows, {len(fi['columns'])} columns)")
        parts.append(f"  Columns: {fi['columns']}")
        for col in fi['columns']:
            dtype = fi['dtypes'].get(col, '?')
            samp = fi['samples'].get(col, [])
            parts.append(f"    {col} ({dtype}): {samp}")

    # Multimodal file inventory
    if modality and modality.primary != "tabular":
        parts.append("")
        parts.append("=== Non-Tabular Data ===")
        parts.append(f"Primary modality: {modality.primary}")
        parts.append(f"Total files: {modality.total_file_count}")

        # File counts by extension
        parts.append("File type counts:")
        for ext, count in sorted(modality.file_counts_by_ext.items(), key=lambda x: -x[1]):
            parts.append(f"  {ext}: {count}")

        # Directory tree
        if modality.dir_tree:
            parts.append("")
            parts.append("Directory structure:")
            parts.append(modality.dir_tree)

        # Sample file info for images
        if modality.has_images:
            parts.append("")
            parts.append("Sample image files:")
            for fpath in sample_files(modality.image_files, n=5):
                info = get_file_basic_info(fpath)
                if "error" in info:
                    parts.append(f"  {info['name']}: {info['error']}")
                else:
                    parts.append(
                        f"  {info['name']}: {info.get('width', '?')}x{info.get('height', '?')} "
                        f"{info.get('mode', '?')} ({info['size_bytes']} bytes)"
                    )

        # Sample file info for audio
        if modality.has_audio:
            parts.append("")
            parts.append("Sample audio files:")
            for fpath in sample_files(modality.audio_files, n=5):
                info = get_file_basic_info(fpath)
                if "error" in info:
                    parts.append(f"  {info['name']}: {info['error']}")
                else:
                    parts.append(
                        f"  {info['name']}: {info.get('sample_rate', '?')}Hz, "
                        f"{info.get('channels', '?')}ch, {info.get('duration_sec', '?')}s "
                        f"({info['size_bytes']} bytes)"
                    )

    return "\n".join(parts)


def infer_task_config(
    llm: LLMClient,
    description: str,
    file_inventory: str,
    modality: DataModality = None,
) -> dict:
    """
    Use LLM to infer target column, task type, and metric from description + data.

    Falls back to safe defaults if LLM fails.
    """
    modality_hint = ""
    if modality and modality.primary != "tabular":
        modality_hint = (
            f"\nIMPORTANT: The primary data modality is '{modality.primary}'. "
            "If the data is primarily images or audio (not tabular), set target to null. "
            "The task_type can be 'classification', 'regression', 'segmentation', "
            "'object_detection', 'speech_recognition', or other appropriate types."
        )

    system_prompt = (
        "You are a data science assistant. Given a task description and data file inventory, "
        "infer the prediction target column, task type, and evaluation metric."
        + modality_hint
    )
    user_prompt = f"""Based on the description and data files below, determine:
1. target — the exact column name to predict (must match a column in the data). Set to null if there is no tabular target column (e.g. for image/audio tasks).
2. task_type — "classification", "regression", "segmentation", "object_detection", "speech_recognition", or other
3. metric — the evaluation metric (e.g. "auc", "f1", "rmse", "accuracy", "map", "wer")
4. description — a 1-2 sentence summary of the prediction task

=== Task Description ===
{description[:3000]}

=== Data File Inventory ===
{file_inventory[:3000]}

Return your answer as JSON."""

    result = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=_TASK_INFERENCE_SCHEMA,
    )

    if result["success"] and isinstance(result["content"], dict):
        config = result["content"]
        print(f"[INIT] LLM inferred task config:")
        print(f"  target={config.get('target')}, task_type={config.get('task_type')}, metric={config.get('metric')}")
        return config

    # Fallback: best-effort defaults
    print("[INIT] WARNING: LLM task inference failed, using defaults")
    return {
        "target": None if (modality and modality.primary != "tabular") else "",
        "task_type": "auto",
        "metric": "auto",
        "description": description[:500] if description else "No description provided",
    }


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
    verbose: bool = True,
    max_turns_per_layer: int = 3,
    stop_threshold: float = 0.85,
    layer_timeout_sec: int = 180,
) -> bool:
    """
    Run the complete data profiling pipeline.

    Args:
        data_dir: Path to directory containing description.md + data files.
        output_dir: Directory for outputs (report, plots, etc.).
        llm_provider: LLM provider ("openai" or "anthropic").
        llm_model: Model name to use.
        verbose: Whether to print progress messages.
        max_turns_per_layer: Max retries/turns for each layer agent loop.
        stop_threshold: Score threshold used to converge a layer early.
        layer_timeout_sec: Per-layer timeout budget.

    Returns:
        True if pipeline completed successfully.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Data Profiler Pipeline")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Initialize LLM
    llm_config = LLMConfig(provider=llm_provider, model=llm_model)
    llm = LLMClient(llm_config)

    # Discover data directory
    print("[INIT] Discovering data directory...")
    discovery = discover_data_dir(data_dir)
    description = discovery["description"]
    files_info = discovery["files"]
    primary_data_path = discovery["primary_data_path"]
    modality = discovery["modality"]

    print(f"[INIT] Modality: {modality.primary} "
          f"(tabular={modality.has_tabular}, images={modality.has_images}, audio={modality.has_audio})")
    print(f"[INIT] Total files: {modality.total_file_count}, "
          f"extensions: {dict(sorted(modality.file_counts_by_ext.items(), key=lambda x: -x[1])[:5])}")

    if not primary_data_path and modality.primary == "tabular":
        print("[ERROR] No CSV/Parquet files found in data directory")
        return False

    file_inventory = build_file_inventory_text(files_info, modality)
    if primary_data_path:
        print(f"[INIT] Primary tabular file: {primary_data_path.name}")
    else:
        print(f"[INIT] No primary tabular file (non-tabular data only)")
    if description:
        print(f"[INIT] Description loaded ({len(description)} chars)")
    else:
        print("[INIT] WARNING: No description.md found")

    # Infer task config from description + data
    print("[INIT] Inferring task configuration...")
    task_config = infer_task_config(llm, description, file_inventory, modality)
    task_config["dataset"] = primary_data_path.name if primary_data_path else f"({modality.primary} data)"
    task_config.setdefault("description", description[:500])
    # Attach the file inventory and modality for downstream layers
    task_config["_file_inventory"] = file_inventory
    task_config["_modality_info"] = modality.to_dict()
    task_config["_data_dir"] = str(data_dir)
    print()

    # Initialize core components
    sandbox = CodeSandbox(output_dir / "artifacts", data_dir=data_dir)
    state = StateContext()
    state.data_modality = modality.primary
    loop_config = AgentLoopConfig(
        max_turns_per_layer=max_turns_per_layer,
        stop_threshold=stop_threshold,
        layer_timeout_sec=layer_timeout_sec,
    )

    # Load primary tabular data into sandbox (if any)
    if primary_data_path:
        print("[INIT] Loading tabular dataset...")
        success, message = sandbox.load_dataframe(primary_data_path, max_rows=MAX_ROWS)
        if not success:
            if modality.primary == "tabular":
                print(f"[ERROR] Failed to load data: {message}")
                return False
            else:
                print(f"[INIT] WARNING: Could not load tabular data: {message}")
        else:
            print(f"[INIT] {message}")
    else:
        print("[INIT] No tabular data to load")

    # Setup multimodal support if needed
    if modality.primary != "tabular":
        print("[INIT] Setting up multimodal sandbox (GPU 2)...")
        sandbox.setup_multimodal(gpu_id=2)
        sandbox.inject_helpers()

    print()

    # Execute L0-L3 pipeline
    layers = [
        ("L0", L0Cleaner(sandbox, llm, loop_config=loop_config, verbose=verbose)),
        ("L1", L1Explorer(sandbox, llm, loop_config=loop_config, verbose=verbose)),
        ("L2", L2Aligner(sandbox, llm, loop_config=loop_config, verbose=verbose)),
        ("L3", L3Strategist(sandbox, llm, loop_config=loop_config, verbose=verbose)),
    ]

    for layer_name, layer in layers:
        print("-" * 40)
        try:
            success = layer.execute(state, task_config)

            if not success:
                print(f"[WARNING] {layer_name} completed with issues")

        except Exception as e:
            print(f"[ERROR] {layer_name} failed: {e}")
            state.record_error(layer_name, "exception", str(e))

        print()

    # Generate outputs
    print("-" * 40)
    print("[OUTPUT] Generating final outputs...")

    # Save state
    state_path = output_dir / "pipeline_state.json"
    state.save(state_path)
    print(f"[OUTPUT] State saved to: {state_path}")

    # Save agent trace
    agent_trace_path = output_dir / "agent_trace.json"
    state.save_agent_trace(agent_trace_path)
    print(f"[OUTPUT] Agent trace saved to: {agent_trace_path}")

    # Generate report
    report_gen = ReportGenerator(llm)
    report_path = output_dir / "mle_report.md"
    report_gen.generate(state, task_config, report_path)

    # Generate layer diagnostics
    diagnostics_path = output_dir / "layer_diagnostics.md"
    generate_layer_diagnostics(state, diagnostics_path)

    # Generate execution trace
    trace_path = output_dir / "execution_trace.md"
    generate_execution_trace(state, trace_path)

    # Generate preprocessing script
    script_path = output_dir / "preprocess_pipeline.py"
    generate_preprocessing_script(state, script_path)

    print()
    print("=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Report: {report_path}")
    print(f"Preprocessing Script: {script_path}")
    print(f"Artifacts: {output_dir / 'artifacts'}")
    print()

    # Summary statistics
    print("Summary:")
    print(f"  - Modality: {modality.primary}")
    print(f"  - Task: target={task_config.get('target', '?')}, type={task_config.get('task_type', '?')}, metric={task_config.get('metric', '?')}")
    print(f"  - L0 cleaning actions: {len(state.l0_cleaning_actions)}")
    print(f"  - L1 hypotheses: {len(state.l1_hypotheses)} ({sum(1 for h in state.l1_hypotheses if h.get('confirmed'))} confirmed)")
    print(f"  - L2 domain priors: {len(state.l2_domain_priors)} ({sum(1 for p in state.l2_domain_priors if p.get('confirmed'))} confirmed)")
    print(f"  - L2 selected features: {len(state.l2_selected_features)}")
    print(f"  - L3 model hypotheses: {len(state.l3_hypotheses)}")
    print(f"  - L3 recommended models: {len(state.l3_strategy.get('recommended_models', []))}")
    print(f"  - Layer reports: {list(state.layer_reports.keys())}")
    print(f"  - Errors/warnings: {len(state.errors)}")
    print(f"  - Layer status: {json.dumps(state.layer_status)}")

    has_failed_layers = any(status == "failed" for status in state.layer_status.values())
    return len(state.errors) == 0 and not has_failed_layers


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data Profiler - Automated ML Data Analysis Pipeline"
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to data directory (containing description.md + CSV/Parquet files)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider (default: openai)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model name (default: gpt-4o)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--max-turns-per-layer",
        type=int,
        default=3,
        help="Max agent turns per layer (default: 3)",
    )
    parser.add_argument(
        "--stop-threshold",
        type=float,
        default=0.85,
        help="Convergence threshold for layer score (default: 0.85)",
    )
    parser.add_argument(
        "--layer-timeout-sec",
        type=int,
        default=180,
        help="Per-layer timeout in seconds (default: 180)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    if not args.data_dir.is_dir():
        print(f"Error: {args.data_dir} is not a directory")
        sys.exit(1)

    # Run pipeline
    success = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output,
        llm_provider=args.provider,
        llm_model=args.model,
        verbose=args.verbose,
        max_turns_per_layer=args.max_turns_per_layer,
        stop_threshold=args.stop_threshold,
        layer_timeout_sec=args.layer_timeout_sec,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
