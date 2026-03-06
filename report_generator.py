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

        # Format hypotheses with quality scores, sorted by quality
        sorted_hypotheses = sorted(
            hypotheses,
            key=lambda h: h.quality.overall if h.quality else 0,
            reverse=True,
        )

        hyp_text = "\n".join(
            f"- [{h.conclusion}]{' [' + h.effect_size_grade + ']' if h.effect_size_grade else ''}"
            f"{' (quality=' + f'{h.quality.overall:.2f}' + ')' if h.quality else ''} "
            f"{h.hypothesis}\n  Evidence: {h.evidence_summary}"
            + (f"\n  Effect size: {h.effect_size_detail}" if h.effect_size_detail else "")
            + (f"\n  Suggested features: {', '.join(h.suggested_features)}" if h.suggested_features else "")
            + (f"\n  Skeptical review: {'Agreed' if h.skeptical_review.agrees_with_conclusion else 'Disagreed — ' + '; '.join(h.skeptical_review.concerns)}"
               if h.skeptical_review else "")
            for h in sorted_hypotheses
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

        # Use LLM to extract structured recommendations
        feature_recommendations = self._extract_features_via_llm(hypotheses, constraints)
        rejected_features = self._extract_rejected_via_llm(hypotheses)
        modeling_recommendations = self._extract_modeling_via_llm(
            hypotheses, micro_results.get("patterns", []),
            macro_results.get("findings", []), constraints
        )

        report = DataReport(
            task_overview=macro_results.get("preview", ""),
            macro_findings=macro_results.get("findings", []),
            micro_patterns=micro_results.get("patterns", []),
            hypotheses=hypotheses,
            feature_recommendations=feature_recommendations,
            modeling_recommendations=modeling_recommendations,
            constraints=constraints,
            rejected_features=rejected_features,
        )

        # Save outputs
        self.save(report, report_md)

        return report

    def _extract_features_via_llm(self, hypotheses: List[Hypothesis],
                                   constraints: TaskConstraints = None) -> List[str]:
        """Use LLM to extract prioritized feature recommendations from confirmed hypotheses."""
        confirmed = [h for h in hypotheses if h.conclusion == "confirmed"]
        if not confirmed:
            return []

        hyp_text = "\n".join(
            f"- {h.hypothesis}\n"
            f"  Effect size: {h.effect_size_grade or 'N/A'} ({h.effect_size_detail or 'N/A'})\n"
            f"  Quality score: {f'{h.quality.overall:.2f}' if h.quality else 'N/A'}\n"
            f"  Suggested features: {', '.join(h.suggested_features) if h.suggested_features else 'None'}"
            for h in confirmed
        )

        metric = constraints.evaluation_metric if constraints else "unknown"

        prompt = (
            f"Based on these CONFIRMED hypotheses, generate a prioritized list of feature engineering recommendations.\n\n"
            f"Evaluation metric: {metric}\n\n"
            f"Confirmed hypotheses:\n{hyp_text}\n\n"
            f"Requirements:\n"
            f"- Rank features by priority (considering effect size and quality score)\n"
            f"- Each recommendation should be specific and implementable\n"
            f"- Include the effect size grade in parentheses\n"
            f"- Group related features together\n\n"
            f"Return a JSON array of strings, each being one feature recommendation, ordered by priority.\n"
            f"Example: [\"(strong) Create binary feature for text length > 200 chars\", \"(weak) Add word count feature\"]"
        )

        result = self.llm.chat_json(prompt)
        if result and isinstance(result, list):
            return [str(f) for f in result]
        return []

    def _extract_rejected_via_llm(self, hypotheses: List[Hypothesis]) -> List[str]:
        """Use LLM to summarize ALL rejected hypotheses as anti-recommendations."""
        rejected = [h for h in hypotheses if h.conclusion == "rejected"]
        if not rejected:
            return []

        hyp_text = "\n".join(
            f"- {h.hypothesis}\n  Evidence: {h.evidence_summary}"
            for h in rejected
        )

        prompt = (
            f"These hypotheses were REJECTED by statistical verification. "
            f"For each one, generate a concise anti-recommendation explaining why this direction should be avoided.\n\n"
            f"Rejected hypotheses:\n{hyp_text}\n\n"
            f"Return a JSON array of strings. Each string should start with '[NOT RECOMMENDED]' and briefly explain "
            f"why this direction failed.\n"
            f"Example: [\"[NOT RECOMMENDED] Using comment_karma as predictor — no significant correlation with target (p=0.45)\"]"
        )

        result = self.llm.chat_json(prompt)
        if result and isinstance(result, list):
            return [str(f) for f in result]
        return []

    def _extract_modeling_via_llm(self, hypotheses: List[Hypothesis],
                                   patterns: List[MicroPattern],
                                   findings: List[str],
                                   constraints: TaskConstraints = None) -> List[str]:
        """Use LLM to generate data-driven modeling recommendations."""
        confirmed_count = sum(1 for h in hypotheses if h.conclusion == "confirmed")
        rejected_count = sum(1 for h in hypotheses if h.conclusion == "rejected")

        findings_text = "\n".join(findings[:10]) if findings else "None"
        patterns_text = "\n".join(
            f"- {p.pattern} ({p.confidence})" for p in patterns[:10]
        ) if patterns else "None"

        constraints_text = ""
        if constraints:
            constraints_text = (
                f"Evaluation metric: {constraints.evaluation_metric}\n"
                f"Task type: {constraints.task_type}"
            )

        prompt = (
            f"Based on all analysis results, generate specific, data-driven modeling recommendations.\n\n"
            f"Task constraints:\n{constraints_text or 'Unknown'}\n\n"
            f"Key findings:\n{findings_text}\n\n"
            f"Micro patterns:\n{patterns_text}\n\n"
            f"Hypothesis summary: {confirmed_count} confirmed, {rejected_count} rejected, "
            f"{len(hypotheses) - confirmed_count - rejected_count} inconclusive\n\n"
            f"Requirements:\n"
            f"- Recommendations must be specific to THIS dataset (not generic advice)\n"
            f"- Consider the evaluation metric when suggesting optimization strategies\n"
            f"- Address data challenges discovered (class imbalance, missing values, text processing, etc.)\n"
            f"- Suggest specific model types or techniques based on the data characteristics found\n\n"
            f"Return a JSON array of strings, each being one modeling recommendation."
        )

        result = self.llm.chat_json(prompt)
        if result and isinstance(result, list):
            return [str(r) for r in result]
        return []

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
                    "quality": {
                        "relevance": h.quality.relevance,
                        "novelty": h.quality.novelty,
                        "verifiability": h.quality.verifiability,
                        "actionability": h.quality.actionability,
                        "evidence_strength": h.quality.evidence_strength,
                        "overall": h.quality.overall,
                    } if h.quality else None,
                    "skeptical_review": {
                        "agrees_with_conclusion": h.skeptical_review.agrees_with_conclusion,
                        "reviewer_conclusion": h.skeptical_review.reviewer_conclusion,
                        "concerns": h.skeptical_review.concerns,
                        "final_conclusion": h.skeptical_review.final_conclusion,
                    } if h.skeptical_review else None,
                }
                for h in report.hypotheses
            ],
            "feature_recommendations": report.feature_recommendations,
            "rejected_features": report.rejected_features,
            "modeling_recommendations": report.modeling_recommendations,
        }
        json_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False))
        logger.info(f"Saved JSON report to {json_path}")
