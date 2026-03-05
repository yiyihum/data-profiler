"""Multi-turn agent loop runtime for each pipeline layer."""

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, Optional

from core.state import StateContext


@dataclass
class AgentLoopConfig:
    """Runtime controls for each layer-level agent loop."""

    max_turns_per_layer: int = 3
    stop_threshold: float = 0.85
    layer_timeout_sec: int = 180


class AgentLoop:
    """
    Generic layer loop with turn-level logging, scoring, and fallback.

    A layer provides:
    - step_fn: execute one turn
    - evaluator_fn: score turn output (0.0 - 1.0)
    - apply_fn: persist a successful turn output into state
    - fallback_fn: produce degraded output when no turn reaches threshold
    """

    def __init__(
        self,
        layer_name: str,
        config: AgentLoopConfig,
        state: StateContext,
        verbose: bool = True,
    ) -> None:
        self.layer_name = layer_name
        self.config = config
        self.state = state
        self.verbose = verbose

    def run(
        self,
        objective: str,
        step_fn: Callable[[int, Optional[Dict[str, Any]]], Dict[str, Any]],
        evaluator_fn: Callable[[Dict[str, Any], int], Dict[str, Any]],
        apply_fn: Callable[[Dict[str, Any]], None],
        fallback_fn: Callable[[str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute the layer loop and return a status summary."""
        start_ts = time.monotonic()
        best_score = -1.0
        best_output: Optional[Dict[str, Any]] = None
        best_turn = 0
        no_gain_rounds = 0
        prev_output: Optional[Dict[str, Any]] = None

        for turn in range(1, self.config.max_turns_per_layer + 1):
            elapsed = time.monotonic() - start_ts
            if elapsed > self.config.layer_timeout_sec:
                reason = f"Layer timeout after {elapsed:.1f}s"
                break

            plan_text = (
                f"Turn {turn}: pursue objective with iterative improvement"
                if turn == 1
                else f"Turn {turn}: retry based on previous observation"
            )
            self.state.record_agent_decision(
                self.layer_name,
                {
                    "turn": turn,
                    "objective": objective,
                    "plan": plan_text,
                    "previous_best_score": round(max(best_score, 0.0), 4),
                },
            )

            try:
                output = step_fn(turn, prev_output)
            except Exception as exc:  # pragma: no cover - defensive path
                output = {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "observation": "Unhandled exception in step function",
                }

            eval_result = evaluator_fn(output, turn)
            score = float(eval_result.get("score", 0.0))
            done = bool(eval_result.get("done", False))
            decision = eval_result.get("decision", "continue")
            observation = output.get("observation") or eval_result.get("observation", "")
            error = output.get("error")

            self.state.record_layer_turn(
                self.layer_name,
                {
                    "turn": turn,
                    "success": bool(output.get("success", False)),
                    "score": round(score, 4),
                    "decision": decision,
                    "observation": str(observation),
                    "error": str(error) if error else None,
                    "code_snippets": output.get("code_snippets", []),
                    "llm_interactions": output.get("llm_interactions", []),
                },
            )

            if output.get("success", False) and score > best_score:
                if score - best_score < 0.01:
                    no_gain_rounds += 1
                else:
                    no_gain_rounds = 0
                best_score = score
                best_output = output
                best_turn = turn
            else:
                no_gain_rounds += 1

            if output.get("success", False) and score >= self.config.stop_threshold:
                apply_fn(output)
                self.state.set_layer_status(self.layer_name, "success")
                return {
                    "status": "success",
                    "turns_used": turn,
                    "best_turn": turn,
                    "best_score": score,
                    "reason": "threshold_reached",
                }

            if done:
                break

            if no_gain_rounds >= 2 and best_output is not None:
                break

            prev_output = output
        else:
            reason = "max_turns_reached"

        if best_output is not None:
            apply_fn(best_output)
            self.state.set_layer_status(
                self.layer_name,
                "degraded",
                details={
                    "reason": f"best_score_below_threshold ({best_score:.3f})",
                    "best_turn": best_turn,
                },
            )
            return {
                "status": "degraded",
                "turns_used": self.config.max_turns_per_layer,
                "best_turn": best_turn,
                "best_score": best_score,
                "reason": f"best_score_below_threshold ({best_score:.3f})",
            }

        fallback = fallback_fn(reason)
        fallback_status = fallback.get("status", "degraded")
        fallback_reason = fallback.get("reason", reason)
        self.state.set_layer_status(
            self.layer_name,
            fallback_status,
            details={"reason": fallback_reason},
        )
        return {
            "status": fallback_status,
            "turns_used": self.config.max_turns_per_layer,
            "best_turn": 0,
            "best_score": 0.0,
            "reason": fallback_reason,
        }

    @staticmethod
    def _truncate(value: str, max_len: int) -> str:
        """Clamp diagnostic blobs to avoid oversized state payloads."""
        if len(value) <= max_len:
            return value
        return value[: max_len - 3] + "..."
