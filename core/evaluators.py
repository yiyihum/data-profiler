"""Turn evaluators for each layer in the multi-turn agent loop."""

from typing import Any, Dict


def _decision(score: float, success: bool) -> str:
    if success and score >= 0.85:
        return "converged"
    if not success:
        return "retry_after_error"
    return "continue"


def evaluate_l0_turn(step_output: Dict[str, Any], turn: int) -> Dict[str, Any]:
    """Score L0 output quality (ultra-conservative cleaning + report section)."""
    score = 0.0
    if step_output.get("discovery_success"):
        score += 0.10
    if step_output.get("exploration_success"):
        score += 0.10
    if step_output.get("code_gen_success"):
        score += 0.15
    if step_output.get("code_exec_success"):
        score += 0.20
    if step_output.get("report_section_success"):
        score += 0.25
    if step_output.get("json_synthesis_success"):
        score += 0.10
    if step_output.get("clean_df_ready"):
        score += 0.10

    success = bool(step_output.get("success", False))
    return {
        "score": min(score, 1.0),
        "done": False,
        "decision": _decision(score, success),
        "observation": step_output.get("observation", ""),
    }


def evaluate_l1_turn(step_output: Dict[str, Any], turn: int) -> Dict[str, Any]:
    """Score L1 output quality (hypothesis-verify pattern)."""
    score = 0.0
    if step_output.get("discovery_success"):
        score += 0.08
    if step_output.get("exploration_success"):
        score += 0.07
    if step_output.get("hypothesis_gen_success"):
        score += 0.15
    if step_output.get("verification_success"):
        score += 0.20
    if step_output.get("judgment_success"):
        score += 0.15
    if step_output.get("confirmed_findings_count", 0) > 0:
        score += 0.15
    if step_output.get("report_section_success"):
        score += 0.20

    success = bool(step_output.get("success", False))
    return {
        "score": min(score, 1.0),
        "done": False,
        "decision": _decision(score, success),
        "observation": step_output.get("observation", ""),
    }


def evaluate_l2_turn(step_output: Dict[str, Any], turn: int) -> Dict[str, Any]:
    """Score L2 output quality (domain-prior-verify-apply pattern)."""
    score = 0.0
    if step_output.get("discovery_success"):
        score += 0.05
    if step_output.get("exploration_success"):
        score += 0.05
    if step_output.get("prior_gen_success"):
        score += 0.10
    if step_output.get("verification_success"):
        score += 0.15
    if step_output.get("judgment_success"):
        score += 0.10
    if step_output.get("apply_success"):
        score += 0.15
    if step_output.get("json_synthesis_success"):
        score += 0.10
    if step_output.get("report_section_success"):
        score += 0.15
    if step_output.get("selected_features_count", 0) > 0:
        score += 0.10
    if step_output.get("final_df_ready"):
        score += 0.05

    success = bool(step_output.get("success", False))
    return {
        "score": min(score, 1.0),
        "done": False,
        "decision": _decision(score, success),
        "observation": step_output.get("observation", ""),
    }


def evaluate_l3_turn(step_output: Dict[str, Any], turn: int) -> Dict[str, Any]:
    """Score L3 output quality (AutoML hypotheses)."""
    score = 0.0
    if step_output.get("discovery_success"):
        score += 0.08
    if step_output.get("exploration_success"):
        score += 0.07
    if step_output.get("hypothesis_gen_success"):
        score += 0.30
    n_hypotheses = step_output.get("n_hypotheses", 0)
    if n_hypotheses >= 3:
        score += 0.15
    elif n_hypotheses >= 1:
        score += 0.05
    if step_output.get("report_section_success"):
        score += 0.25
    if step_output.get("validation_strategy_present"):
        score += 0.15

    success = bool(step_output.get("success", False))
    return {
        "score": min(score, 1.0),
        "done": False,
        "decision": _decision(score, success),
        "observation": step_output.get("observation", ""),
    }
