import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from code_executor import CodeExecutor
from config import Config
from data_loader import DataLoader
from llm_client import LLMClient
from models import Hypothesis, HypothesisQuality, SkepticalReview, TaskConstraints

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"


class MacroAnalyzer:
    def __init__(self, llm: LLMClient, loader: DataLoader, executor: CodeExecutor,
                 config: Config):
        self.llm = llm
        self.loader = loader
        self.executor = executor
        self.config = config

    def analyze(self) -> Dict:
        """Phase 1: Macro statistical analysis."""
        logger.info("=== Phase 1: Macro Analysis ===")

        # Step 1a: Extract TaskConstraints from description
        constraints = self._extract_task_constraints()
        logger.info(f"Task type: {constraints.task_type}, Metric: {constraints.evaluation_metric}")
        if constraints.unavailable_fields:
            logger.info(f"Unavailable fields: {constraints.unavailable_fields}")
        if constraints.leakage_risk_fields:
            logger.info(f"Leakage risk fields: {constraints.leakage_risk_fields}")

        # Step 1b: Get data preview
        preview = self.loader.get_preview()
        description = self.loader.description
        target_col = self.loader.detect_target_column()
        logger.info(f"Target column: {target_col}")

        # Step 2: Generate hypotheses (pass constraints context)
        prompt_template = (PROMPT_DIR / "macro_hypothesis_gen.txt").read_text()
        constraints_text = self._format_constraints(constraints)
        prompt = prompt_template.format(
            description=description,
            preview=preview,
            constraints=constraints_text,
        )

        raw_hypotheses = self.llm.chat_json(prompt)
        if not raw_hypotheses:
            logger.warning("Failed to generate macro hypotheses")
            raw_hypotheses = []

        # Step 3: Verify each hypothesis (3-stage pipeline)
        hypotheses = []
        for h_data in raw_hypotheses[:self.config.max_hypotheses]:
            h = Hypothesis(
                hypothesis=h_data.get("hypothesis", ""),
                source="macro",
                related_features=h_data.get("related_features", []),
                hypothesis_type=h_data.get("hypothesis_type", ""),
            )
            h = self._verify_hypothesis(h, target_col, constraints, hypotheses)
            hypotheses.append(h)
            grade_tag = f" [{h.effect_size_grade}]" if h.effect_size_grade else ""
            quality_tag = f" (q={h.quality.overall:.2f})" if h.quality else ""
            logger.info(f"  Hypothesis: {h.hypothesis[:80]}... → {h.conclusion}{grade_tag}{quality_tag}")

        # Step 4: Compile results
        macro_findings = []
        for h in hypotheses:
            grade_tag = f" ({h.effect_size_grade})" if h.effect_size_grade else ""
            if h.conclusion == "confirmed":
                macro_findings.append(f"[CONFIRMED{grade_tag}] {h.hypothesis}: {h.evidence_summary}")
            elif h.conclusion == "rejected":
                macro_findings.append(f"[REJECTED] {h.hypothesis}: {h.evidence_summary}")
            else:
                macro_findings.append(f"[INCONCLUSIVE] {h.hypothesis}: {h.evidence_summary}")

        return {
            "target_col": target_col,
            "preview": preview,
            "hypotheses": hypotheses,
            "findings": macro_findings,
            "constraints": constraints,
        }

    def _extract_task_constraints(self) -> TaskConstraints:
        """Step 1a: Parse description.md to extract structured task constraints."""
        description = self.loader.description
        train_columns = ", ".join(self.loader.data.columns.tolist())

        # Load test data columns if available
        test_columns = "Not available"
        if self.config.test_data_path:
            try:
                test_path = self.config.test_data_path
                import pandas as pd
                if test_path.endswith(".json"):
                    import json as json_mod
                    with open(test_path) as f:
                        raw = json_mod.load(f)
                    test_df = pd.DataFrame(raw)
                elif test_path.endswith(".csv"):
                    test_df = pd.read_csv(test_path)
                elif test_path.endswith(".parquet"):
                    test_df = pd.read_parquet(test_path)
                else:
                    test_df = None

                if test_df is not None:
                    test_columns = ", ".join(test_df.columns.tolist())
                    # Auto-detect fields missing in test
                    train_set = set(self.loader.data.columns)
                    test_set = set(test_df.columns)
                    auto_unavailable = list(train_set - test_set)
                    if auto_unavailable:
                        logger.info(f"Auto-detected unavailable fields (train-only): {auto_unavailable}")
            except Exception as e:
                logger.warning(f"Failed to load test data: {e}")

        prompt_template = (PROMPT_DIR / "task_description_analysis.txt").read_text()
        prompt = prompt_template.format(
            description=description,
            train_columns=train_columns,
            test_columns=test_columns,
        )

        result = self.llm.chat_json(prompt)
        if not result or not isinstance(result, dict):
            logger.warning("Failed to extract task constraints, using defaults")
            return TaskConstraints()

        constraints = TaskConstraints(
            evaluation_metric=result.get("evaluation_metric", ""),
            task_type=result.get("task_type", ""),
            unavailable_fields=result.get("unavailable_fields", []),
            leakage_risk_fields=result.get("leakage_risk_fields", []),
            special_notes=result.get("special_notes", []),
        )

        # Merge auto-detected unavailable fields
        if self.config.test_data_path:
            try:
                train_set = set(self.loader.data.columns)
                test_set = set(test_df.columns) if test_df is not None else set()
                auto_unavailable = list(train_set - test_set)
                for f in auto_unavailable:
                    if f not in constraints.unavailable_fields:
                        constraints.unavailable_fields.append(f)
            except Exception:
                pass

        return constraints

    @staticmethod
    def _format_constraints(constraints: TaskConstraints) -> str:
        """Format TaskConstraints as readable text for prompt injection."""
        lines = []
        if constraints.evaluation_metric:
            lines.append(f"- Evaluation Metric: {constraints.evaluation_metric}")
        if constraints.task_type:
            lines.append(f"- Task Type: {constraints.task_type}")
        if constraints.unavailable_fields:
            lines.append(f"- Unavailable Fields (not in test): {', '.join(constraints.unavailable_fields)}")
        if constraints.leakage_risk_fields:
            lines.append(f"- Leakage Risk Fields: {', '.join(constraints.leakage_risk_fields)}")
        if constraints.special_notes:
            for note in constraints.special_notes:
                lines.append(f"- Note: {note}")
        return "\n".join(lines) if lines else "No specific constraints extracted."

    def _verify_hypothesis(self, hypothesis: Hypothesis, target_col: str,
                           constraints: TaskConstraints = None,
                           existing_hypotheses: list = None) -> Hypothesis:
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
                previous_error = f"\nPrevious attempt failed with error:\n{result.stderr}\nPlease fix the code."
                logger.warning(f"  Verification attempt {attempt + 1} failed: {result.stderr[:100]}")

        if not execution_success:
            hypothesis.conclusion = "inconclusive"
            hypothesis.evidence_summary = "Verification code failed after all retries"
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
            hypothesis.evidence_summary = "Failed to parse analysis result"

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
                relevance=0.5, novelty=0.5, verifiability=1.0,
                actionability=0.5, evidence_strength=0.5
            )

        # Gate feature suggestions: only confirmed + quality > 0.5
        if hypothesis.conclusion != "confirmed" or hypothesis.quality.overall <= 0.5:
            hypothesis.suggested_features = []

        return hypothesis

    @staticmethod
    def _clean_code(code: str) -> str:
        """Remove markdown code block markers."""
        import re
        match = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return code.strip()
