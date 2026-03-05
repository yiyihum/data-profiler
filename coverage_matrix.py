import logging
from typing import Dict, List, Tuple

from models import Hypothesis

logger = logging.getLogger(__name__)


class HypothesisCoverageMatrix:
    HYPOTHESIS_TYPES = [
        "distribution",
        "target_correlation",
        "interaction",
        "text_pattern",
        "missing_pattern",
        "temporal_pattern",
    ]

    # Which hypothesis types apply to which feature types
    TYPE_APPLICABILITY = {
        "numeric": ["distribution", "target_correlation", "interaction", "missing_pattern"],
        "categorical": ["distribution", "target_correlation", "interaction", "missing_pattern"],
        "text": ["text_pattern", "target_correlation", "interaction"],
        "datetime": ["temporal_pattern", "target_correlation", "distribution"],
    }

    def __init__(self, features: List[str], feature_types: Dict[str, str]):
        self.features = features
        self.feature_types = feature_types
        self.matrix: Dict[Tuple[str, str], List[Hypothesis]] = {}
        self._valid_cells = self._compute_valid_cells()

    def _compute_valid_cells(self) -> List[Tuple[str, str]]:
        """Compute all valid (feature, hypothesis_type) combinations."""
        cells = []
        for feat in self.features:
            ft = self.feature_types.get(feat, "numeric")
            applicable = self.TYPE_APPLICABILITY.get(ft, ["distribution", "target_correlation"])
            for ht in applicable:
                cells.append((feat, ht))
        return cells

    def register_hypothesis(self, hypothesis: Hypothesis, features: List[str],
                           hypothesis_type: str):
        """Register a hypothesis into the coverage matrix."""
        for f in features:
            if f not in self.features:
                continue
            key = (f, hypothesis_type)
            self.matrix.setdefault(key, []).append(hypothesis)
        logger.debug(f"Registered hypothesis for {features} × {hypothesis_type}")

    def get_uncovered_cells(self) -> List[Tuple[str, str]]:
        """Return (feature, type) combinations not covered by any hypothesis."""
        return [cell for cell in self._valid_cells if cell not in self.matrix]

    def get_weakly_covered_cells(self, min_count: int = 1) -> List[Tuple[str, str]]:
        """Return combinations with insufficient coverage."""
        weak = []
        for cell in self._valid_cells:
            hyps = self.matrix.get(cell, [])
            if len(hyps) < min_count:
                weak.append(cell)
        return weak

    def generate_coverage_prompt(self) -> str:
        """Generate a description of coverage status for LLM prompting."""
        lines = []
        covered = []
        uncovered = []

        for cell in self._valid_cells:
            feat, ht = cell
            hyps = self.matrix.get(cell, [])
            if hyps:
                confirmed = sum(1 for h in hyps if h.conclusion == "confirmed")
                rejected = sum(1 for h in hyps if h.conclusion == "rejected")
                covered.append(
                    f"  ✓ {feat} × {ht}: {len(hyps)} hypotheses "
                    f"({confirmed} confirmed, {rejected} rejected)"
                )
            else:
                uncovered.append(f"  ✗ {feat} × {ht}")

        lines.append("## Covered Areas:")
        lines.extend(covered[:20])  # Limit output length
        if len(covered) > 20:
            lines.append(f"  ... and {len(covered) - 20} more")

        lines.append("\n## Uncovered Areas (need hypotheses):")
        lines.extend(uncovered[:30])
        if len(uncovered) > 30:
            lines.append(f"  ... and {len(uncovered) - 30} more")

        return "\n".join(lines)

    def get_coverage_stats(self) -> dict:
        """Return coverage statistics."""
        total = len(self._valid_cells)
        covered = sum(1 for cell in self._valid_cells if cell in self.matrix)
        all_hyps = [h for hyps in self.matrix.values() for h in hyps]
        return {
            "total_valid_cells": total,
            "covered_cells": covered,
            "coverage_rate": covered / total if total > 0 else 0,
            "total_hypotheses": len(all_hyps),
            "confirmed_hypotheses": sum(1 for h in all_hyps if h.conclusion == "confirmed"),
            "rejected_hypotheses": sum(1 for h in all_hyps if h.conclusion == "rejected"),
            "inconclusive_hypotheses": sum(1 for h in all_hyps if h.conclusion == "inconclusive"),
        }
