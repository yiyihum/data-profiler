from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskConstraints:
    """Structured constraints extracted from task description."""
    evaluation_metric: str = ""              # e.g. "AUC-ROC"
    task_type: str = ""                      # "classification" | "regression" | ...
    unavailable_fields: List[str] = field(default_factory=list)   # fields missing in test
    special_notes: List[str] = field(default_factory=list)        # other important constraints


@dataclass
class MicroObservation:
    """LLM analysis result for a single data point."""
    data_point_id: str
    label: Any
    observation: str
    key_patterns: List[str]
    raw_data_snippet: str = ""  # Original data point text/feature values for citation


@dataclass
class MicroPattern:
    """Pattern summarized from multiple MicroObservations."""
    pattern: str
    evidence: List[str]           # data point IDs
    confidence: str               # high / medium / low
    source: str                   # "contrastive" | "single_class" | "cross_feature"
    evidence_snippets: List[str] = field(default_factory=list)  # concrete data citations


@dataclass
class HypothesisQuality:
    """Multi-dimensional quality score for a hypothesis (0-1)."""
    relevance: float          # relevance to task / evaluation metric
    novelty: float            # whether it duplicates existing findings
    verifiability: float      # whether verification code executed successfully
    actionability: float      # whether it can be directly converted to a feature
    evidence_strength: float  # statistical evidence strength (effect size)

    @property
    def overall(self) -> float:
        """Weighted overall score."""
        return (self.relevance * 0.25 + self.novelty * 0.15 +
                self.verifiability * 0.15 + self.actionability * 0.25 +
                self.evidence_strength * 0.20)


@dataclass
class SkepticalReview:
    """Independent review result from LLM_3 Skeptical Reviewer."""
    agrees_with_conclusion: bool    # whether reviewer agrees with LLM_2's conclusion
    reviewer_conclusion: str        # reviewer's own conclusion
    concerns: List[str]             # issues found during review
    final_conclusion: str           # final conclusion


@dataclass
class Hypothesis:
    """A verifiable hypothesis."""
    hypothesis: str
    source: str  # "macro" | "micro" | "bridge" | "coverage_gap"
    related_features: List[str] = field(default_factory=list)
    hypothesis_type: str = ""  # distribution, target_correlation, etc.
    micro_pattern: Optional[MicroPattern] = None
    verification_code: str = ""
    execution_result: str = ""
    conclusion: str = ""          # "confirmed" | "rejected" | "inconclusive"
    evidence_summary: str = ""
    suggested_features: List[str] = field(default_factory=list)
    effect_size_grade: str = ""   # "strong" | "weak" | "none" | "" (for qualitative)
    effect_size_detail: str = ""  # e.g. "Cohen's d=0.8, p<0.001"
    quality: Optional[HypothesisQuality] = None
    skeptical_review: Optional[SkepticalReview] = None


@dataclass
class DataReport:
    """Final data profiling report."""
    task_overview: str
    macro_findings: List[str]
    micro_patterns: List[MicroPattern]
    hypotheses: List[Hypothesis]
    feature_recommendations: List[str]
    modeling_recommendations: List[str]
    constraints: Optional[TaskConstraints] = None
    rejected_features: List[str] = field(default_factory=list)  # features from rejected hypotheses
