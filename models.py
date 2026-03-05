from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskConstraints:
    """Structured constraints extracted from task description."""
    evaluation_metric: str = ""              # e.g. "AUC-ROC"
    task_type: str = ""                      # "classification" | "regression" | ...
    unavailable_fields: List[str] = field(default_factory=list)   # fields missing in test
    leakage_risk_fields: List[str] = field(default_factory=list)  # fields that may leak target
    special_notes: List[str] = field(default_factory=list)        # other important constraints


@dataclass
class MicroObservation:
    """单条数据点的 LLM 分析结果"""
    data_point_id: str
    label: Any
    observation: str
    key_patterns: List[str]
    raw_data_snippet: str = ""  # Original data point text/feature values for citation


@dataclass
class MicroPattern:
    """从多条 MicroObservation 归纳出的模式"""
    pattern: str
    evidence: List[str]           # data point IDs
    confidence: str               # high / medium / low
    source: str                   # "contrastive" | "single_class" | "cross_feature"
    evidence_snippets: List[str] = field(default_factory=list)  # concrete data citations


@dataclass
class Hypothesis:
    """可验证的假设"""
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


@dataclass
class DataReport:
    """最终数据分析报告"""
    task_overview: str
    macro_findings: List[str]
    micro_patterns: List[MicroPattern]
    hypotheses: List[Hypothesis]
    feature_recommendations: List[str]
    modeling_recommendations: List[str]
    constraints: Optional[TaskConstraints] = None
    rejected_features: List[str] = field(default_factory=list)  # features from rejected hypotheses
