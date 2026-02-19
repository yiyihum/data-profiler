from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from acp_framework.agents.coder_agent import CoderAgent
from acp_framework.agents.data_agent import DataAgent
from acp_framework.agents.hypothesor_agent import HypothesorAgent
from acp_framework.agents.planner_agent import PlannerAgent
from acp_framework.core.consolidator import RuleConsolidator, ValidatedRule
from acp_framework.core.data_state import DataState
from acp_framework.core.sandbox import SandboxConfig, SandboxRunner
from acp_framework.tools.skills_library import SkillsLibrary


@dataclass(slots=True)
class PipelineResult:
    rules: list[ValidatedRule]
    profile: dict[str, Any]
    coverage_ratio: float


def run_acp_pipeline(
    raw_df: pd.DataFrame,
    target_col: str | None = None,
    text_col: str | None = None,
    *,
    max_rounds: int = 12,
    sample_size: int = 40,
    llm_model: str = "gpt-4o-mini",
    api_key_env: str = "OPENAI_API_KEY",
    llm_enabled: bool = True,
    mock_mode: bool = False,
    sandbox_mode: str = "dev_local",
    skills_db_path: str | Path | None = None,
) -> PipelineResult:
    data_state = DataState(raw_df, target_col=target_col, text_col=text_col)

    planner = PlannerAgent(max_rounds=max_rounds)
    data_agent = DataAgent(data_state)
    hypothesor = HypothesorAgent(
        model=llm_model,
        api_key_env=api_key_env,
        enabled=llm_enabled,
        mock_mode=mock_mode,
    )
    coder = CoderAgent(
        model=llm_model,
        api_key_env=api_key_env,
        enabled=llm_enabled,
        mock_mode=mock_mode,
    )
    sandbox = SandboxRunner(SandboxConfig(mode=sandbox_mode))
    consolidator = RuleConsolidator()
    skills = SkillsLibrary(db_path=skills_db_path)

    final_profile_rules: list[ValidatedRule] = []
    accepted_codes: set[str] = set()

    for round_id in range(max_rounds + 1):
        strategy_plan = planner.decide_next_step(
            coverage=data_state.coverage_ratio,
            current_rules=final_profile_rules,
            round_id=round_id,
        )

        if strategy_plan.intent == "STOP":
            print(f"Planner: STOP ({strategy_plan.reason})")
            break

        print(
            f"\n[Round {round_id + 1}] intent={strategy_plan.intent} "
            f"strategy={strategy_plan.sampling_strategy} coverage={data_state.coverage_ratio:.2%}"
        )

        view_df = data_agent.get_view(
            strategy_plan.sampling_strategy,
            sample_size=sample_size,
            random_state=42 + round_id,
        )
        if view_df.empty:
            print("DataAgent: no rows for this strategy")
            continue

        skills_context = skills.render_prompt_context(strategy_plan.intent, max_items=5)
        hypotheses = hypothesor.propose(
            view_df,
            strategy_plan.intent,
            max_hypotheses=strategy_plan.max_hypotheses,
            skills_context=skills_context,
        )
        if not hypotheses:
            print("Hypothesor: no hypotheses generated")
            continue

        candidates = coder.generate_batch(hypotheses, context=strategy_plan.intent, skills_context=skills_context)
        if not candidates:
            print("Coder: no executable candidates generated")
            continue

        valid_rules = consolidator.validate_and_consolidate(
            candidates=candidates,
            data_state=data_state,
            intent=strategy_plan.intent,
            sandbox=sandbox,
        )

        if not valid_rules:
            print("Consolidator: no rules passed validation")
            continue

        accepted_in_round = 0
        for rule in valid_rules:
            code_key = rule.code.strip()
            if code_key in accepted_codes:
                continue

            incremental_mask = np.logical_and(rule.mask, ~data_state.coverage_mask)
            if not incremental_mask.any():
                continue

            data_state.update_coverage(rule.mask)
            final_profile_rules.append(rule)
            accepted_codes.add(code_key)
            skills.register(rule)
            accepted_in_round += 1
            print(
                f"  [+] accepted {rule.id} score={rule.score:.4f} "
                f"coverage={rule.hit_rate:.2%} {rule.description}"
            )

        if accepted_in_round == 0:
            print("Consolidator: all candidates were duplicates or had zero incremental coverage")
            continue

        print(f"Global coverage ratio: {data_state.coverage_ratio:.2%}")

    skills.flush()

    profile = build_profile(data_state, final_profile_rules)
    return PipelineResult(rules=final_profile_rules, profile=profile, coverage_ratio=data_state.coverage_ratio)


def build_profile(data_state: DataState, rules: list[ValidatedRule]) -> dict[str, Any]:
    directives: list[dict[str, Any]] = []
    for idx, rule in enumerate(sorted(rules, key=lambda item: item.score, reverse=True), start=1):
        directives.append(
            {
                "id": f"rule_{idx:03d}",
                "intent": rule.intent,
                "description": rule.description,
                "score": rule.score,
                "hit_rate": rule.hit_rate,
                "source": rule.source,
                "code": rule.code,
            }
        )

    return {
        "meta": {
            "rows": data_state.total_rows,
            "text_col": data_state.text_col,
            "target_col": data_state.target_col,
            "final_coverage_ratio": data_state.coverage_ratio,
        },
        "strategy": {
            "difficulty": "Easy" if data_state.coverage_ratio >= 0.8 else "Medium",
            "recommended_model": "LightGBM + Rule Features",
        },
        "directives": directives,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACP State-Driven Multi-Agent Pipeline")
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--target-col", default=None, help="Target label column")
    parser.add_argument("--text-col", default=None, help="Text feature column")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--sandbox-mode", choices=["dev_local", "strict_firejail"], default="dev_local")
    parser.add_argument("--skills-db", default="./skills_library.json")
    parser.add_argument("--out", default="./profile.json")
    parser.add_argument("--mock-llm", action="store_true", help="Use deterministic mock hypotheses/rules")
    return parser.parse_args()


def _detect_target_column(df: pd.DataFrame) -> str | None:
    for col in ("label", "target", "y", "class", "insult", "Insult"):
        if col in df.columns:
            return col
    return None


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Input data file not found: {data_path}")

    df = pd.read_csv(data_path)
    target_col = args.target_col or _detect_target_column(df)

    result = run_acp_pipeline(
        df,
        target_col=target_col,
        text_col=args.text_col,
        max_rounds=args.max_rounds,
        sample_size=args.sample_size,
        llm_model=args.model,
        api_key_env=args.api_key_env,
        llm_enabled=not args.mock_llm,
        mock_mode=args.mock_llm,
        sandbox_mode=args.sandbox_mode,
        skills_db_path=args.skills_db,
    )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.profile, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone. final_coverage_ratio={result.coverage_ratio:.2%}")
    print(f"Profile written to: {out_path}")


if __name__ == "__main__":
    main()
