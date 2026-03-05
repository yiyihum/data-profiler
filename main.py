#!/usr/bin/env python3
"""Micro-Macro Data Understanding Framework for LLM-based AutoML."""

import argparse
import logging
import sys

from code_executor import CodeExecutor
from config import Config
from data_loader import DataLoader
from hypothesis_engine import HypothesisEngine
from llm_client import LLMClient
from macro_analyzer import MacroAnalyzer
from micro_analyzer import MicroAnalyzer
from report_generator import ReportGenerator


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(config: Config):
    """Run the full Micro-Macro data profiling pipeline."""
    logger = logging.getLogger("main")
    logger.info("Starting Micro-Macro Data Profiling Framework")
    logger.info(f"Data: {config.data_path}")
    logger.info(f"Model: {config.model_name} @ {config.vllm_endpoint}")

    # Initialize components
    client = LLMClient(config)
    loader = DataLoader(config)
    executor = CodeExecutor(config)

    # Phase 1: Macro Analysis (now includes TaskConstraints extraction)
    macro = MacroAnalyzer(client, loader, executor, config)
    macro_results = macro.analyze()
    constraints = macro_results.get("constraints")
    logger.info(f"Phase 1 complete: {len(macro_results['hypotheses'])} hypotheses verified")

    # Phase 2: Micro Analysis
    target_col = macro_results["target_col"]
    task_context = macro_results["preview"][:1000]
    micro = MicroAnalyzer(client, loader, executor, config)
    micro_results = micro.analyze(target_col, task_context)
    logger.info(f"Phase 2 complete: {len(micro_results['patterns'])} patterns discovered")

    # Phase 3: Bridge (Micro → Macro with Coverage Matrix)
    engine = HypothesisEngine(client, loader, executor, config)
    bridge_results = engine.bridge(
        micro_results, macro_results,
        micro_analyzer=micro,
        constraints=constraints,
    )
    logger.info(f"Phase 3 complete: {len(bridge_results['hypotheses'])} total hypotheses")

    # Phase 4: Report
    reporter = ReportGenerator(client, config)
    report = reporter.generate(
        macro_results, micro_results, bridge_results,
        constraints=constraints,
    )
    logger.info(f"Phase 4 complete: report saved to {config.output_dir}")

    # Print summary
    stats = bridge_results.get("coverage_stats", {})
    confirmed = stats.get("confirmed_hypotheses", 0)
    rejected = stats.get("rejected_hypotheses", 0)
    total = stats.get("total_hypotheses", 0)
    coverage = stats.get("coverage_rate", 0)
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"  Hypotheses: {total} total, {confirmed} confirmed, {rejected} rejected")
    logger.info(f"  Coverage: {coverage:.1%}")
    logger.info(f"  Feature recommendations: {len(report.feature_recommendations)}")
    if report.rejected_features:
        logger.info(f"  Rejected features: {len(report.rejected_features)}")
    if constraints:
        logger.info(f"  Evaluation metric: {constraints.evaluation_metric}")
        logger.info(f"  Unavailable fields: {len(constraints.unavailable_fields)}")
        logger.info(f"  Leakage risk fields: {len(constraints.leakage_risk_fields)}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Micro-Macro Data Understanding Framework"
    )
    parser.add_argument("--data-path", required=True, help="Path to data file (JSON/CSV)")
    parser.add_argument("--test-data-path", default="", help="Path to test data file (for field comparison)")
    parser.add_argument("--description-path", default="", help="Path to task description (Markdown)")
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000/v1",
                       help="vLLM API endpoint")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name on vLLM")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--micro-sample-per-class", type=int, default=10)
    parser.add_argument("--max-micro-rounds", type=int, default=3)
    parser.add_argument("--max-hypotheses", type=int, default=15)
    parser.add_argument("--max-bridge-iterations", type=int, default=2)
    parser.add_argument("--code-timeout", type=int, default=60)
    parser.add_argument("--firejail", action="store_true", help="Enable firejail sandboxing")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)

    config = Config(
        vllm_endpoint=args.vllm_endpoint,
        model_name=args.model_name,
        data_path=args.data_path,
        test_data_path=args.test_data_path,
        description_path=args.description_path,
        micro_sample_per_class=args.micro_sample_per_class,
        max_micro_rounds=args.max_micro_rounds,
        max_hypotheses=args.max_hypotheses,
        max_bridge_iterations=args.max_bridge_iterations,
        code_timeout=args.code_timeout,
        output_dir=args.output_dir,
        firejail_enabled=args.firejail,
    )
    main(config)
