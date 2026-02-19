from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from agents.coder_agent import CoderAgent
from agents.ingestion_agent import IngestionAgent
from agents.sampler_agent import SamplerAgent, SamplerConfig
from core.compiler import ProfileCompiler
from core.engine import CDTEngine, EngineConfig
from core.memory import MetadataStore
from core.sandbox import build_sandbox_runner
from core.workspace import create_run_workspace, stage_user_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACP MVP Runner")
    parser.add_argument("--data", type=str, default=None, help="Input data path (file or directory)")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Path to YAML config")
    parser.add_argument("--workspace-root", type=str, default=None, help="Workspace root override")
    parser.add_argument("--max-depth", type=int, default=None, help="Engine max depth override")
    parser.add_argument("--samples-per-node", type=int, default=None, help="Samples per node override")
    parser.add_argument("--candidates-per-node", type=int, default=None, help="Candidates per node override")
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def choose_input_path(args: argparse.Namespace, runtime, default_input_path: str | None) -> Path:
    selected_input = args.data or default_input_path
    if not selected_input:
        raise ValueError("No input path provided. Pass --data or set runtime.default_input_path in config.")

    staged = stage_user_input(selected_input, runtime, dst_name="user_data")
    src = Path(selected_input).resolve()
    if src.is_file():
        return staged / src.name
    return staged


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    runtime_cfg = config.get("runtime", {})
    sandbox_cfg = config.get("sandbox", {})
    llm_cfg = config.get("llm", {})
    engine_cfg_raw = config.get("engine", {})

    workspace_root = args.workspace_root or runtime_cfg.get("workspace_root", "./runs")
    runtime = create_run_workspace(workspace_root)
    configure_logging(runtime.logs_dir / "run.log")

    logging.info("run_id=%s workspace=%s", runtime.run_id, runtime.workspace_dir)

    default_input_path = runtime_cfg.get("default_input_path")
    input_path = choose_input_path(args, runtime, default_input_path)
    logging.info("Input path staged at workspace: %s", input_path)

    (runtime.artifacts_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    llm_enabled = bool(llm_cfg.get("enabled", True))
    llm_api_key_env = str(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
    if llm_enabled and not os.getenv(llm_api_key_env):
        logging.error("LLM is enabled but API key environment variable is missing: %s", llm_api_key_env)
        raise SystemExit(2)

    try:
        sandbox = build_sandbox_runner(
            tmp_dir=runtime.tmp_dir,
            timeout_s=int(sandbox_cfg.get("timeout_s", 5)),
            project_root=Path(__file__).resolve().parent,
            mode=str(sandbox_cfg.get("mode", "strict_firejail")),
            firejail_bin=str(sandbox_cfg.get("firejail_bin", "firejail")),
            rlimit_as_mb=int(sandbox_cfg.get("rlimit_as_mb", 4096)),
        )
    except RuntimeError as exc:
        logging.error("%s", exc)
        raise SystemExit(2) from exc
    logging.info("Sandbox runner initialized: %s", type(sandbox).__name__)

    ingestion = IngestionAgent(
        sandbox_runner=sandbox,
        default_task_description=str(runtime_cfg.get("default_task_description", "analyze the dataset")),
        task_description_max_chars=int(runtime_cfg.get("task_description_max_chars", 20_000)),
        task_summary_max_chars=int(runtime_cfg.get("task_summary_max_chars", 2_000)),
    )
    ingestion_result = ingestion.run(input_path, runtime.artifacts_dir)
    logging.info("Ingestion done: parquet=%s shape=%s", ingestion_result.parquet_path, ingestion_result.dataset_shape)
    logging.info(
        "Task description source=%s path=%s",
        ingestion_result.metadata.get("task_description_source", "unknown"),
        ingestion_result.metadata.get("task_description_path"),
    )

    sampler = SamplerAgent(SamplerConfig(random_seed=int(runtime_cfg.get("random_seed", 42))))
    coder = CoderAgent(
        llm_enabled=llm_enabled,
        llm_provider=str(llm_cfg.get("provider", "openai")),
        llm_model=str(llm_cfg.get("model", "gpt-4o-mini")),
        llm_api_key_env=llm_api_key_env,
        llm_temperature=float(llm_cfg.get("temperature", 0.2)),
        llm_timeout_s=int(llm_cfg.get("timeout_s", 30)),
    )
    memory = MetadataStore()

    engine_config = EngineConfig(
        max_depth=int(args.max_depth if args.max_depth is not None else engine_cfg_raw.get("max_depth", 3)),
        samples_per_node=int(
            args.samples_per_node if args.samples_per_node is not None else engine_cfg_raw.get("samples_per_node", 5)
        ),
        candidates_per_node=int(
            args.candidates_per_node
            if args.candidates_per_node is not None
            else engine_cfg_raw.get("candidates_per_node", 5)
        ),
        purity_threshold=float(engine_cfg_raw.get("purity_threshold", 0.98)),
        min_samples_split=int(engine_cfg_raw.get("min_samples_split", 40)),
        timeout_s=int(sandbox_cfg.get("timeout_s", 5)),
        embedding_backend=str(config.get("embedding", {}).get("backend", "auto")),
    )

    engine = CDTEngine(sampler=sampler, coder=coder, sandbox=sandbox, memory=memory)
    engine_result = engine.run(
        parquet_path=str(ingestion_result.parquet_path),
        config=engine_config,
        metadata=ingestion_result.metadata,
    )

    tree_path = runtime.artifacts_dir / "tree.json"
    tree_path.write_text(json.dumps(engine_result.tree, ensure_ascii=False, indent=2), encoding="utf-8")

    compiler = ProfileCompiler()
    profile_path = runtime.artifacts_dir / "profile.json"
    profile = compiler.compile(engine_result, profile_path)

    summary = {
        "run_id": runtime.run_id,
        "workspace": str(runtime.workspace_dir),
        "input": str(input_path),
        "parquet": str(ingestion_result.parquet_path),
        "tree": str(tree_path),
        "profile": str(profile_path),
        "rules": len(profile.get("directives", [])),
    }
    summary_path = runtime.artifacts_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info("Run complete: %s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
