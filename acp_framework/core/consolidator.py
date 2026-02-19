from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from acp_framework.core.data_state import DataState
from acp_framework.core.sandbox import SandboxExecutionError, SandboxRunner


@dataclass(slots=True)
class ValidatedRule:
    id: str
    description: str
    code: str
    mask: np.ndarray
    score: float
    hit_rate: float
    intent: str
    source: str = "llm"


class RuleConsolidator:
    """Validate generated rules, score them, and remove near-duplicates."""

    def __init__(
        self,
        jaccard_threshold: float = 0.85,
        min_coverage: float = 0.005,
        max_coverage: float = 0.95,
        feature_max_coverage: float = 0.40,
        anomaly_max_coverage: float = 0.05,
        min_ig: float = 0.01,
        min_label_shift: float = 0.03,
    ) -> None:
        self.jaccard_threshold = float(jaccard_threshold)
        self.min_coverage = float(min_coverage)
        self.max_coverage = float(max_coverage)
        self.feature_max_coverage = float(feature_max_coverage)
        self.anomaly_max_coverage = float(anomaly_max_coverage)
        self.min_ig = float(min_ig)
        self.min_label_shift = float(min_label_shift)

    def validate_and_consolidate(
        self,
        candidates: Sequence[Any],
        data_state: DataState,
        intent: str,
        sandbox: SandboxRunner,
    ) -> list[ValidatedRule]:
        text_values = data_state.get_text_values()
        staged: list[ValidatedRule] = []

        for idx, candidate in enumerate(candidates, start=1):
            description = _candidate_value(candidate, "description", default=f"candidate_{idx}")
            code = _candidate_value(candidate, "code", default="")
            source = _candidate_value(candidate, "source", default="llm")

            if "def check(" not in code:
                continue

            try:
                mask = sandbox.execute_rule(code, text_values)
            except SandboxExecutionError:
                continue

            if mask.shape[0] != data_state.total_rows:
                continue

            hit_rate = float(mask.sum() / max(len(mask), 1))
            if not (self.min_coverage <= hit_rate <= self.max_coverage):
                continue

            score = self._score_rule(data_state, intent, mask, hit_rate)
            if score is None:
                continue

            staged.append(
                ValidatedRule(
                    id=f"rule_{len(staged) + 1:03d}",
                    description=description,
                    code=code,
                    mask=mask,
                    score=score,
                    hit_rate=hit_rate,
                    intent=intent,
                    source=source,
                )
            )

        staged.sort(key=lambda item: item.score, reverse=True)
        unique: list[ValidatedRule] = []
        for rule in staged:
            duplicate = any(_jaccard_similarity(rule.mask, kept.mask) > self.jaccard_threshold for kept in unique)
            if not duplicate:
                unique.append(rule)

        for idx, rule in enumerate(unique, start=1):
            rule.id = f"rule_{idx:03d}"

        return unique

    def _score_rule(self, data_state: DataState, intent: str, mask: np.ndarray, hit_rate: float) -> float | None:
        normalized_intent = (intent or "feature_mining").strip().lower()

        if normalized_intent == "anomaly_detection":
            if hit_rate > self.anomaly_max_coverage:
                return None
            return self.calc_isolation_score(data_state, mask)

        if data_state.target_col and data_state.target_col in data_state.df.columns:
            if hit_rate > self.feature_max_coverage:
                return None

            labels = data_state.df[data_state.target_col]
            ig = self.calc_ig(labels, mask)
            if ig < self.min_ig:
                return None

            label_shift = self.calc_label_shift(labels, mask)
            if label_shift < self.min_label_shift:
                return None

            coverage_penalty = max(0.4, 1.0 - 0.5 * (hit_rate / max(self.feature_max_coverage, 1e-6)))
            return float(ig * (1.0 + label_shift) * coverage_penalty)

        # Unsupervised fallback: prefer non-trivial coverage around 10%
        return max(0.0, 1.0 - abs(hit_rate - 0.10))

    @staticmethod
    def calc_ig(labels: pd.Series, mask: np.ndarray) -> float:
        y = labels.fillna("__NULL__")
        left = y[mask]
        right = y[~mask]

        if left.empty or right.empty:
            return 0.0

        base_entropy = _entropy(y)
        left_entropy = _entropy(left)
        right_entropy = _entropy(right)

        w_left = len(left) / len(y)
        w_right = len(right) / len(y)
        conditional_entropy = w_left * left_entropy + w_right * right_entropy
        return float(max(base_entropy - conditional_entropy, 0.0))

    @staticmethod
    def calc_isolation_score(data_state: DataState, mask: np.ndarray) -> float:
        text_series = data_state.df[data_state.text_col].fillna("").astype(str)
        lengths = text_series.str.len().to_numpy(dtype=float)

        selected = lengths[mask]
        remaining = lengths[~mask]
        if selected.size == 0 or remaining.size == 0:
            return 0.0

        mean_gap = abs(float(selected.mean() - remaining.mean()))
        std_all = float(lengths.std() + 1e-6)
        separation = min(mean_gap / std_all, 5.0) / 5.0

        hit_rate = float(mask.mean())
        rarity = max(0.0, 1.0 - abs(hit_rate - 0.02) / 0.02)
        return float(0.7 * separation + 0.3 * rarity)

    @staticmethod
    def calc_label_shift(labels: pd.Series, mask: np.ndarray) -> float:
        all_counts = labels.fillna("__NULL__").astype(str).value_counts(normalize=True, dropna=False)
        hit = labels[mask].fillna("__NULL__").astype(str)
        if hit.empty:
            return 0.0
        hit_counts = hit.value_counts(normalize=True, dropna=False)

        idx = all_counts.index.union(hit_counts.index)
        p = all_counts.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
        q = hit_counts.reindex(idx, fill_value=0.0).to_numpy(dtype=float)

        # Total variation distance, range [0, 1]
        return float(0.5 * np.abs(p - q).sum())


def _candidate_value(candidate: Any, field: str, default: str) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get(field, default))
    return str(getattr(candidate, field, default))


def _entropy(values: pd.Series) -> float:
    counts = values.value_counts(normalize=True, dropna=False)
    probs = counts.to_numpy(dtype=float)
    safe = probs[probs > 0]
    return float(-(safe * np.log2(safe)).sum())


def _jaccard_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)
