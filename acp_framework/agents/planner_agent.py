from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True)
class StrategyPlan:
    intent: str
    sampling_strategy: str
    max_hypotheses: int = 3
    reason: str = ""


class PlannerAgent:
    """A state-machine planner that drives the coverage loop."""

    def __init__(
        self,
        target_coverage: float = 0.92,
        max_rounds: int = 12,
        min_progress: float = 0.002,
        max_stalled_rounds: int = 3,
    ) -> None:
        self.target_coverage = float(target_coverage)
        self.max_rounds = int(max_rounds)
        self.min_progress = float(min_progress)
        self.max_stalled_rounds = int(max_stalled_rounds)

        self._last_coverage: float | None = None
        self._stalled_rounds = 0

    def decide_next_step(self, coverage: float, current_rules: Sequence[object], round_id: int) -> StrategyPlan:
        if round_id >= self.max_rounds:
            return StrategyPlan(intent="STOP", sampling_strategy="none", reason="Reached max rounds")

        if coverage >= self.target_coverage:
            return StrategyPlan(intent="STOP", sampling_strategy="none", reason="Coverage target reached")

        if self._last_coverage is not None:
            progress = coverage - self._last_coverage
            if progress < self.min_progress:
                self._stalled_rounds += 1
            else:
                self._stalled_rounds = 0
        self._last_coverage = coverage

        if self._stalled_rounds >= self.max_stalled_rounds and coverage > 0.85:
            return StrategyPlan(intent="STOP", sampling_strategy="none", reason="Coverage progress stalled")

        if coverage < 0.10:
            return StrategyPlan(
                intent="feature_mining",
                sampling_strategy="random_baseline",
                max_hypotheses=4,
                reason="Bootstrap broad patterns",
            )

        if coverage < 0.75:
            return StrategyPlan(
                intent="feature_mining",
                sampling_strategy="uncovered_only",
                max_hypotheses=3,
                reason="Focus on uncovered hard cases",
            )

        return StrategyPlan(
            intent="anomaly_detection",
            sampling_strategy="extreme_lengths",
            max_hypotheses=2,
            reason="Mine edge-case anomalies",
        )
