import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from config import Config
from llm_client import LLMClient
from models import DataReport, Hypothesis, MicroPattern, TaskConstraints

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"


class ReportGenerator:
    def __init__(self, llm: LLMClient, config: Config):
        self.llm = llm
        self.config = config

    def generate(self, macro_results: Dict, micro_results: Dict,
                 bridge_results: Dict,
                 constraints: TaskConstraints = None) -> DataReport:
        """Phase 4: Generate comprehensive data profiling report."""
        logger.info("=== Phase 4: Report Generation ===")

        prompt_template = (PROMPT_DIR / "report_synthesis.txt").read_text()

        # Format inputs
        macro_text = "\n".join(macro_results.get("findings", []))
        micro_pattern_text = "\n".join(
            f"- [{p.confidence}] {p.pattern} (source: {p.source})"
            + (f"\n  Evidence: {'; '.join(p.evidence_snippets[:3])}" if p.evidence_snippets else "")
            for p in micro_results.get("patterns", [])
        )
        hypotheses = bridge_results.get("hypotheses", [])

        # Format hypotheses with effect size grades
        hyp_text = "\n".join(
            f"- [{h.conclusion}]{' [' + h.effect_size_grade + ']' if h.effect_size_grade else ''} "
            f"{h.hypothesis}\n  Evidence: {h.evidence_summary}"
            + (f"\n  Effect size: {h.effect_size_detail}" if h.effect_size_detail else "")
            + (f"\n  Suggested features: {', '.join(h.suggested_features)}" if h.suggested_features else "")
            for h in hypotheses
        )
        coverage_stats = json.dumps(bridge_results.get("coverage_stats", {}), indent=2)
        sampling_summary = json.dumps(
            micro_results.get("sampling_summary", {}), indent=2, default=str
        )

        # Format constraints
        constraints_text = ""
        if constraints:
            parts = []
            if constraints.evaluation_metric:
                parts.append(f"Evaluation Metric: {constraints.evaluation_metric}")
            if constraints.task_type:
                parts.append(f"Task Type: {constraints.task_type}")
            if constraints.unavailable_fields:
                parts.append(f"Unavailable Fields: {', '.join(constraints.unavailable_fields)}")
            if constraints.leakage_risk_fields:
                parts.append(f"Leakage Risk Fields: {', '.join(constraints.leakage_risk_fields)}")
            if constraints.special_notes:
                parts.append("Special Notes:\n" + "\n".join(f"  - {n}" for n in constraints.special_notes))
            constraints_text = "\n".join(parts)

        prompt = prompt_template.format(
            description=macro_results.get("preview", "")[:2000],
            macro_findings=macro_text,
            micro_patterns=micro_pattern_text,
            hypotheses=hyp_text,
            coverage_stats=coverage_stats,
            sampling_summary=sampling_summary,
            constraints=constraints_text or "None",
        )

        report_md = self.llm.chat(prompt, max_tokens=8192)

        # Build structured report
        confirmed_features = self._extract_features(hypotheses)
        rejected_features = self._extract_rejected_features(hypotheses)

        report = DataReport(
            task_overview=macro_results.get("preview", ""),
            macro_findings=macro_results.get("findings", []),
            micro_patterns=micro_results.get("patterns", []),
            hypotheses=hypotheses,
            feature_recommendations=confirmed_features,
            modeling_recommendations=[],
            constraints=constraints,
            rejected_features=rejected_features,
        )

        # Save outputs
        self.save(report, report_md)

        return report

    def _extract_features(self, hypotheses: List[Hypothesis]) -> List[str]:
        """Extract feature recommendations from CONFIRMED hypotheses only."""
        features = []
        for h in hypotheses:
            if h.conclusion == "confirmed" and h.suggested_features:
                grade = f" ({h.effect_size_grade})" if h.effect_size_grade else ""
                for f in h.suggested_features:
                    entry = f"{f}{grade}"
                    if entry not in features:
                        features.append(entry)
        return features

    def _extract_rejected_features(self, hypotheses: List[Hypothesis]) -> List[str]:
        """Extract features from REJECTED hypotheses as anti-recommendations."""
        rejected = []
        for h in hypotheses:
            if h.conclusion == "rejected" and h.suggested_features:
                for f in h.suggested_features:
                    entry = f"[不建议] {f} — 假设 '{h.hypothesis[:60]}...' 已被拒绝"
                    if entry not in rejected:
                        rejected.append(entry)
        return rejected

    def save(self, report: DataReport, report_md: str):
        """Save report as both Markdown and JSON."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save Markdown report
        md_path = output_dir / "data_profiling_report.md"
        md_path.write_text(report_md)
        logger.info(f"Saved Markdown report to {md_path}")

        # Save JSON report
        json_path = output_dir / "data_profiling_report.json"
        report_dict = {
            "task_overview": report.task_overview,
            "constraints": {
                "evaluation_metric": report.constraints.evaluation_metric,
                "task_type": report.constraints.task_type,
                "unavailable_fields": report.constraints.unavailable_fields,
                "leakage_risk_fields": report.constraints.leakage_risk_fields,
                "special_notes": report.constraints.special_notes,
            } if report.constraints else None,
            "macro_findings": report.macro_findings,
            "micro_patterns": [
                {
                    "pattern": p.pattern,
                    "evidence": p.evidence,
                    "confidence": p.confidence,
                    "source": p.source,
                    "evidence_snippets": p.evidence_snippets,
                }
                for p in report.micro_patterns
            ],
            "hypotheses": [
                {
                    "hypothesis": h.hypothesis,
                    "source": h.source,
                    "conclusion": h.conclusion,
                    "evidence_summary": h.evidence_summary,
                    "suggested_features": h.suggested_features,
                    "related_features": h.related_features,
                    "hypothesis_type": h.hypothesis_type,
                    "effect_size_grade": h.effect_size_grade,
                    "effect_size_detail": h.effect_size_detail,
                }
                for h in report.hypotheses
            ],
            "feature_recommendations": report.feature_recommendations,
            "rejected_features": report.rejected_features,
            "modeling_recommendations": report.modeling_recommendations,
        }
        json_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False))
        logger.info(f"Saved JSON report to {json_path}")
