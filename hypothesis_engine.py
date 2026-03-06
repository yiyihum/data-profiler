import logging
import re
from pathlib import Path
from typing import Dict, List

from code_executor import CodeExecutor
from config import Config
from coverage_matrix import HypothesisCoverageMatrix
from data_loader import DataLoader
from llm_client import LLMClient
from micro_analyzer import MicroAnalyzer
from models import Hypothesis, HypothesisQuality, MicroPattern, SkepticalReview, TaskConstraints

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"

MAX_PER_CELL = 2  # Maximum hypotheses per coverage cell


class HypothesisEngine:
    def __init__(self, llm: LLMClient, loader: DataLoader, executor: CodeExecutor,
                 config: Config):
        self.llm = llm
        self.loader = loader
        self.executor = executor
        self.config = config

    def bridge(self, micro_results: Dict, macro_results: Dict,
               micro_analyzer: MicroAnalyzer = None,
               constraints: TaskConstraints = None) -> Dict:
        """Phase 3: Bridge micro patterns to macro verification with coverage matrix."""
        logger.info("=== Phase 3: Hypothesis Bridge ===")

        target_col = macro_results["target_col"]
        micro_patterns = micro_results["patterns"]
        feature_types = self.loader.get_feature_types()

        # Filter features for coverage matrix (exclude ID/username/text columns)
        # Also exclude unavailable fields from constraints
        unavailable = set(constraints.unavailable_fields) if constraints else set()
        analysis_features = [
            f for f in self.loader.data.columns
            if f != target_col
            and "username" not in f.lower()
            and "request_id" not in f.lower()
            and "giver" not in f.lower()
            and f not in unavailable
        ]

        coverage = HypothesisCoverageMatrix(analysis_features, feature_types)

        # Register macro hypotheses
        for h in macro_results.get("hypotheses", []):
            if h.related_features and h.hypothesis_type:
                coverage.register_hypothesis(h, h.related_features, h.hypothesis_type)

        all_hypotheses = list(macro_results.get("hypotheses", []))

        for iteration in range(self.config.max_bridge_iterations):
            logger.info(f"--- Bridge Iteration {iteration} ---")

            # 1a. Generate hypotheses from micro patterns
            hyps_from_micro = self._hypotheses_from_patterns(
                micro_patterns, macro_results, target_col
            )
            logger.info(f"  Hypotheses from micro patterns: {len(hyps_from_micro)}")

            # 1b. Generate hypotheses from coverage gaps
            uncovered = coverage.get_uncovered_cells()
            hyps_from_gaps = []
            if uncovered:
                hyps_from_gaps = self._hypotheses_from_gaps(
                    coverage, macro_results, target_col
                )
                logger.info(f"  Hypotheses from coverage gaps: {len(hyps_from_gaps)}")

            new_hypotheses = hyps_from_micro + hyps_from_gaps

            # Dedup: remove semantically duplicate hypotheses
            new_hypotheses = self._dedup_hypotheses(new_hypotheses, all_hypotheses)
            logger.info(f"  After dedup: {len(new_hypotheses)} new hypotheses")

            # Filter by max per cell
            new_hypotheses = self._filter_by_cell_limit(new_hypotheses, coverage)

            # 2. Verify each hypothesis (3-stage pipeline)
            for h in new_hypotheses[:self.config.max_hypotheses]:
                h = self._verify_hypothesis(h, target_col, constraints, all_hypotheses)
                coverage.register_hypothesis(h, h.related_features, h.hypothesis_type)
                all_hypotheses.append(h)
                grade_tag = f" [{h.effect_size_grade}]" if h.effect_size_grade else ""
                quality_tag = f" (q={h.quality.overall:.2f})" if h.quality else ""
                logger.info(f"  {h.hypothesis[:60]}... → {h.conclusion}{grade_tag}{quality_tag}")

            # 3. Check coverage
            stats = coverage.get_coverage_stats()
            logger.info(f"  Coverage: {stats['coverage_rate']:.1%} "
                       f"({stats['covered_cells']}/{stats['total_valid_cells']})")

            if stats["coverage_rate"] >= self.config.min_coverage_rate:
                logger.info("  Coverage threshold reached, stopping bridge iterations")
                break

            # 4. Optionally trigger new micro sampling
            if micro_analyzer and micro_analyzer.sampler and iteration < self.config.max_bridge_iterations - 1:
                logger.info("  Triggering additional micro sampling from bridge...")
                new_samples = micro_analyzer.sampler.adaptive_sample(
                    micro_patterns, all_hypotheses
                )
                if len(new_samples) > 0:
                    task_context = macro_results.get("preview", "")[:500]
                    micro_analyzer.analyze_samples(new_samples, target_col, task_context)
                    micro_patterns = micro_analyzer.patterns

        return {
            "hypotheses": all_hypotheses,
            "coverage_stats": coverage.get_coverage_stats(),
            "coverage_prompt": coverage.generate_coverage_prompt(),
        }

    def _dedup_hypotheses(self, new_hypotheses: List[Hypothesis],
                          existing_hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Remove semantically duplicate hypotheses using LLM."""
        if not new_hypotheses or not existing_hypotheses:
            return new_hypotheses

        existing_text = "\n".join(
            f"- {h.hypothesis}" for h in existing_hypotheses
        )

        deduped = []
        # Batch dedup check
        new_text = "\n".join(
            f"{i}: {h.hypothesis}" for i, h in enumerate(new_hypotheses)
        )

        prompt = (
            f"Compare these NEW hypotheses against EXISTING ones. "
            f"Return a JSON array of indices (0-based) of NEW hypotheses that are NOT duplicates "
            f"of any existing hypothesis. Two hypotheses are duplicates if they test essentially "
            f"the same thing about the same features, even if worded differently.\n\n"
            f"EXISTING hypotheses:\n{existing_text}\n\n"
            f"NEW hypotheses:\n{new_text}\n\n"
            f"Return JSON array of non-duplicate indices, e.g. [0, 2, 4]"
        )

        result = self.llm.chat_json(prompt)
        if result and isinstance(result, list):
            for idx in result:
                if isinstance(idx, int) and 0 <= idx < len(new_hypotheses):
                    deduped.append(new_hypotheses[idx])
            return deduped if deduped else new_hypotheses
        return new_hypotheses

    def _filter_by_cell_limit(self, hypotheses: List[Hypothesis],
                               coverage: HypothesisCoverageMatrix) -> List[Hypothesis]:
        """Filter out hypotheses that would exceed max per coverage cell."""
        filtered = []
        for h in hypotheses:
            # Check if any of the target cells already has MAX_PER_CELL hypotheses
            can_add = True
            for feat in h.related_features:
                key = (feat, h.hypothesis_type)
                existing = coverage.matrix.get(key, [])
                if len(existing) >= MAX_PER_CELL:
                    can_add = False
                    break
            if can_add:
                filtered.append(h)
        return filtered

    def _hypotheses_from_patterns(self, patterns: List[MicroPattern],
                                   macro_results: Dict,
                                   target_col: str) -> List[Hypothesis]:
        """Generate testable hypotheses from micro patterns."""
        prompt_template = (PROMPT_DIR / "bridge_hypothesis_gen.txt").read_text()

        pattern_text = "\n".join(
            f"- [{p.confidence}] {p.pattern} (source: {p.source}, "
            f"evidence: {len(p.evidence)} points)"
            + (f"\n  Example evidence: {'; '.join(p.evidence_snippets[:2])}" if p.evidence_snippets else "")
            for p in patterns
        )

        macro_text = "\n".join(macro_results.get("findings", [])[:10])

        prompt = prompt_template.format(
            task_context=macro_results.get("preview", "")[:1000],
            micro_patterns=pattern_text or "No patterns",
            macro_results=macro_text or "No findings",
        )

        result = self.llm.chat_json(prompt)
        hypotheses = []
        if result and isinstance(result, list):
            for h_data in result:
                h = Hypothesis(
                    hypothesis=h_data.get("hypothesis", ""),
                    source="bridge",
                    related_features=h_data.get("related_features", []),
                    hypothesis_type=h_data.get("hypothesis_type", "target_correlation"),
                )
                hypotheses.append(h)
        return hypotheses

    def _hypotheses_from_gaps(self, coverage: HypothesisCoverageMatrix,
                               macro_results: Dict,
                               target_col: str) -> List[Hypothesis]:
        """Generate hypotheses targeting coverage gaps."""
        prompt_template = (PROMPT_DIR / "coverage_gap_hypothesis.txt").read_text()

        coverage_status = coverage.generate_coverage_prompt()
        macro_text = "\n".join(macro_results.get("findings", [])[:10])

        prompt = prompt_template.format(
            task_context=macro_results.get("preview", "")[:1000],
            coverage_status=coverage_status,
            existing_findings=macro_text or "No findings",
        )

        result = self.llm.chat_json(prompt)
        hypotheses = []
        if result and isinstance(result, list):
            for h_data in result:
                h = Hypothesis(
                    hypothesis=h_data.get("hypothesis", ""),
                    source="coverage_gap",
                    related_features=h_data.get("related_features", []),
                    hypothesis_type=h_data.get("hypothesis_type", "target_correlation"),
                )
                hypotheses.append(h)
        return hypotheses

    def _verify_hypothesis(self, hypothesis: Hypothesis, target_col: str,
                           constraints: TaskConstraints = None,
                           existing_hypotheses: List[Hypothesis] = None) -> Hypothesis:
        """Three-stage verification: LLM_1 (code gen) → Execute → LLM_2 (analysis) → LLM_3 (skeptical review)."""
        df = self.loader.data
        columns = ", ".join(df.columns.tolist())
        shape = f"{df.shape[0]} rows × {df.shape[1]} columns"

        code_prompt_template = (PROMPT_DIR / "verification_code_gen.txt").read_text()
        analysis_prompt_template = (PROMPT_DIR / "verification_result_analysis.txt").read_text()
        review_prompt_template = (PROMPT_DIR / "skeptical_review.txt").read_text()
        quality_prompt_template = (PROMPT_DIR / "hypothesis_quality_scoring.txt").read_text()

        # === Stage 1: LLM_1 generates verification code + execute ===
        previous_error = ""
        execution_success = False
        for attempt in range(self.config.max_verification_retries):
            code_prompt = code_prompt_template.format(
                hypothesis=hypothesis.hypothesis,
                approach=hypothesis.hypothesis_type,
                columns=columns,
                target_col=target_col,
                shape=shape,
                previous_error=previous_error,
            )

            code = self.llm.chat(code_prompt, temperature=0.2)
            code = self._clean_code(code)
            hypothesis.verification_code = code

            result = self.executor.execute(code)
            hypothesis.execution_result = result.output

            if result.success:
                execution_success = True
                break
            else:
                previous_error = f"\nPrevious attempt failed:\n{result.stderr}\nFix the code."
                logger.warning(f"    Attempt {attempt + 1} failed: {result.stderr[:100]}")

        if not execution_success:
            hypothesis.conclusion = "inconclusive"
            hypothesis.evidence_summary = "Code execution failed after retries"
            hypothesis.quality = HypothesisQuality(
                relevance=0.0, novelty=0.0, verifiability=0.0,
                actionability=0.0, evidence_strength=0.0
            )
            return hypothesis

        # === Stage 2: LLM_2 analyzes execution output ===
        analysis_prompt = analysis_prompt_template.format(
            hypothesis=hypothesis.hypothesis,
            output=result.output,
        )
        analysis = self.llm.chat_json(analysis_prompt)
        if analysis:
            hypothesis.conclusion = analysis.get("conclusion", "inconclusive")
            hypothesis.evidence_summary = analysis.get("evidence_summary", "")
            hypothesis.suggested_features = analysis.get("suggested_features", [])
            hypothesis.effect_size_grade = analysis.get("effect_size_grade", "")
            hypothesis.effect_size_detail = analysis.get("effect_size_detail", "")
        else:
            hypothesis.conclusion = "inconclusive"
            hypothesis.evidence_summary = "Failed to parse analysis"

        # === Stage 3: LLM_3 Skeptical Reviewer ===
        review_prompt = review_prompt_template.format(
            hypothesis=hypothesis.hypothesis,
            code=hypothesis.verification_code,
            output=result.output,
            initial_conclusion=hypothesis.conclusion,
            initial_evidence=hypothesis.evidence_summary,
        )
        review_result = self.llm.chat_json(review_prompt)
        if review_result and isinstance(review_result, dict):
            review = SkepticalReview(
                agrees_with_conclusion=review_result.get("agrees_with_conclusion", True),
                reviewer_conclusion=review_result.get("reviewer_conclusion", ""),
                concerns=review_result.get("concerns", []),
                final_conclusion=review_result.get("final_conclusion", ""),
            )
            hypothesis.skeptical_review = review

            if not review.agrees_with_conclusion:
                if hypothesis.conclusion == "confirmed":
                    hypothesis.conclusion = "inconclusive"
                hypothesis.evidence_summary += (
                    f"\n[Reviewer concern: {'; '.join(review.concerns)}]"
                )
                logger.info(f"    Skeptical reviewer disagreed → downgraded to {hypothesis.conclusion}")

        # === Quality Scoring ===
        constraints_text = ""
        if constraints:
            constraints_text = (
                f"Metric: {constraints.evaluation_metric}, "
                f"Type: {constraints.task_type}"
            )

        existing_text = "\n".join(
            f"- {h.hypothesis}" for h in (existing_hypotheses or [])
        ) or "None"

        quality_prompt = quality_prompt_template.format(
            hypothesis=hypothesis.hypothesis,
            conclusion=hypothesis.conclusion,
            evidence_summary=hypothesis.evidence_summary,
            effect_size_grade=hypothesis.effect_size_grade or "N/A",
            effect_size_detail=hypothesis.effect_size_detail or "N/A",
            constraints=constraints_text or "None",
            existing_hypotheses=existing_text,
        )
        quality_result = self.llm.chat_json(quality_prompt)
        if quality_result and isinstance(quality_result, dict):
            hypothesis.quality = HypothesisQuality(
                relevance=float(quality_result.get("relevance", 0)),
                novelty=float(quality_result.get("novelty", 0)),
                verifiability=float(quality_result.get("verifiability", 0)),
                actionability=float(quality_result.get("actionability", 0)),
                evidence_strength=float(quality_result.get("evidence_strength", 0)),
            )
        else:
            hypothesis.quality = HypothesisQuality(
                relevance=0.5, novelty=0.5, verifiability=1.0 if execution_success else 0.0,
                actionability=0.5, evidence_strength=0.5
            )

        # Gate feature suggestions: only confirmed + quality > 0.5
        if hypothesis.conclusion != "confirmed" or hypothesis.quality.overall <= 0.5:
            hypothesis.suggested_features = []

        return hypothesis

    @staticmethod
    def _clean_code(code: str) -> str:
        match = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return code.strip()
