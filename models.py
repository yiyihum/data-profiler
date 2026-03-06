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
class HypothesisQuality:
    """假设的多维度质量评分 (0-1)"""
    relevance: float          # 跟任务/评估指标的相关性
    novelty: float            # 是否重复已有发现
    verifiability: float      # 验证代码是否成功执行
    actionability: float      # 能否直接转化为特征
    evidence_strength: float  # 统计证据强度 (效应量)

    @property
    def overall(self) -> float:
        """加权综合分"""
        return (self.relevance * 0.25 + self.novelty * 0.15 +
                self.verifiability * 0.15 + self.actionability * 0.25 +
                self.evidence_strength * 0.20)


@dataclass
class SkepticalReview:
    """LLM_3 Skeptical Reviewer 的独立审查结果"""
    agrees_with_conclusion: bool    # 是否同意 LLM_2 的结论
    reviewer_conclusion: str        # reviewer 自己的结论
    concerns: List[str]             # 审查中发现的问题
    final_conclusion: str           # 最终结论


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
    quality: Optional[HypothesisQuality] = None
    skeptical_review: Optional[SkepticalReview] = None


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
