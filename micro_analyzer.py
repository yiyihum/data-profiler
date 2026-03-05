import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from adaptive_sampler import AdaptiveSampler
from code_executor import CodeExecutor
from config import Config
from data_loader import DataLoader
from llm_client import LLMClient
from models import Hypothesis, MicroObservation, MicroPattern

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"


class MicroAnalyzer:
    def __init__(self, llm: LLMClient, loader: DataLoader, executor: CodeExecutor,
                 config: Config):
        self.llm = llm
        self.loader = loader
        self.executor = executor
        self.config = config
        self.all_observations: List[MicroObservation] = []
        self.patterns: List[MicroPattern] = []
        self.sampler: Optional[AdaptiveSampler] = None

    def analyze(self, target_col: str, task_context: str = "") -> Dict:
        """Phase 2: Micro LLM-driven data point analysis with adaptive sampling."""
        logger.info("=== Phase 2: Micro Analysis ===")

        df = self.loader.data
        self.sampler = AdaptiveSampler(
            df, target_col, self.config, self.llm, self.executor
        )

        for round_num in range(self.config.max_micro_rounds):
            logger.info(f"--- Micro Round {round_num} ---")

            # Step 1: Get samples
            if round_num == 0:
                samples = self.sampler.initial_sample()
            else:
                samples = self.sampler.adaptive_sample(self.patterns, [])

            if len(samples) == 0:
                logger.info("No more samples available, stopping micro analysis")
                break

            # Step 2: Single point analysis
            observations = self._analyze_single_points(samples, target_col, task_context)
            self.all_observations.extend(observations)
            logger.info(f"  Analyzed {len(observations)} individual data points")

            # Step 3: Contrastive analysis
            contrastive_patterns = self._contrastive_analysis(
                observations, target_col, task_context
            )

            # Step 4: Pattern synthesis
            result = self._synthesize_patterns(task_context)
            if result and not result.get("new_patterns_found", True) and round_num > 0:
                logger.info("  No new patterns found, stopping micro analysis")
                break

            logger.info(f"  Total patterns after round {round_num}: {len(self.patterns)}")

        return {
            "observations": self.all_observations,
            "patterns": self.patterns,
            "sampling_summary": self.sampler.get_sampling_summary(),
        }

    def analyze_samples(self, samples: pd.DataFrame, target_col: str,
                       task_context: str = "") -> List[MicroObservation]:
        """Analyze additional samples (called from bridge phase)."""
        observations = self._analyze_single_points(samples, target_col, task_context)
        self.all_observations.extend(observations)
        self._synthesize_patterns(task_context)
        return observations

    def _analyze_single_points(self, samples: pd.DataFrame, target_col: str,
                               task_context: str) -> List[MicroObservation]:
        """Analyze each data point individually."""
        prompt_template = (PROMPT_DIR / "micro_single_point.txt").read_text()
        observations = []

        for idx, row in samples.iterrows():
            data_point_str = self.loader.format_data_point(row)
            label = row.get(target_col, "unknown")

            prompt = prompt_template.format(
                task_context=task_context,
                data_point_id=str(idx),
                label=label,
                data_point=data_point_str,
            )

            result = self.llm.chat_json(prompt)
            if result and isinstance(result, dict):
                obs = MicroObservation(
                    data_point_id=str(idx),
                    label=label,
                    observation=result.get("observation", ""),
                    key_patterns=result.get("key_patterns", []),
                    raw_data_snippet=self._make_snippet(row, target_col),
                )
            elif result and isinstance(result, list) and len(result) > 0:
                first = result[0] if isinstance(result[0], dict) else {"observation": str(result)}
                obs = MicroObservation(
                    data_point_id=str(idx),
                    label=label,
                    observation=first.get("observation", str(result)),
                    key_patterns=first.get("key_patterns", []),
                    raw_data_snippet=self._make_snippet(row, target_col),
                )
            else:
                obs = MicroObservation(
                    data_point_id=str(idx),
                    label=label,
                    observation="Failed to analyze",
                    key_patterns=[],
                    raw_data_snippet=self._make_snippet(row, target_col),
                )
            observations.append(obs)

        return observations

    def _make_snippet(self, row: pd.Series, target_col: str) -> str:
        """Create a concise raw data snippet for citation purposes."""
        parts = []
        label = row.get(target_col, "?")
        parts.append(f"[{target_col}={label}]")
        for col, val in row.items():
            if col == target_col:
                continue
            if isinstance(val, str):
                if len(val) > 100:
                    parts.append(f"{col}: '{val[:100]}...'")
                elif len(val) > 0:
                    parts.append(f"{col}: '{val}'")
            elif pd.notna(val):
                parts.append(f"{col}={val}")
            if len(parts) > 8:  # Keep snippet concise
                parts.append("...")
                break
        return " | ".join(parts)

    def _contrastive_analysis(self, observations: List[MicroObservation],
                              target_col: str, task_context: str) -> List[MicroPattern]:
        """Compare positive vs negative examples."""
        prompt_template = (PROMPT_DIR / "micro_contrastive.txt").read_text()

        # Group by label
        label_groups: Dict[str, List[MicroObservation]] = {}
        for obs in observations:
            key = str(obs.label)
            label_groups.setdefault(key, []).append(obs)

        labels = list(label_groups.keys())
        if len(labels) < 2:
            return []

        # Compare first two label groups
        pos_label, neg_label = labels[0], labels[1]
        # Determine which is positive (True/1) and which is negative
        for l in labels:
            if l in ("True", "1", "true", "yes"):
                pos_label = l
            elif l in ("False", "0", "false", "no"):
                neg_label = l

        pos_obs = label_groups.get(pos_label, [])[:5]
        neg_obs = label_groups.get(neg_label, [])[:5]

        if not pos_obs or not neg_obs:
            return []

        # Include raw data snippets in contrastive analysis
        pos_text = "\n\n".join(
            f"[ID: {o.data_point_id}] {o.observation}\n  Raw: {o.raw_data_snippet}"
            for o in pos_obs
        )
        neg_text = "\n\n".join(
            f"[ID: {o.data_point_id}] {o.observation}\n  Raw: {o.raw_data_snippet}"
            for o in neg_obs
        )

        prompt = prompt_template.format(
            task_context=task_context,
            positive_label=pos_label,
            negative_label=neg_label,
            positive_observations=pos_text,
            negative_observations=neg_text,
        )

        result = self.llm.chat_json(prompt)
        patterns = []
        if result and isinstance(result, list):
            for p in result:
                pattern = MicroPattern(
                    pattern=p.get("pattern", ""),
                    evidence=p.get("evidence", []),
                    confidence=p.get("confidence", "medium"),
                    source="contrastive",
                    evidence_snippets=p.get("evidence_snippets", []),
                )
                patterns.append(pattern)
                self.patterns.append(pattern)

        return patterns

    def _synthesize_patterns(self, task_context: str) -> Optional[dict]:
        """Synthesize all observations into consolidated patterns."""
        prompt_template = (PROMPT_DIR / "micro_pattern_summary.txt").read_text()

        # Include raw data snippets in synthesis for concrete citations
        obs_text = "\n\n".join(
            f"[ID: {o.data_point_id}, Label: {o.label}] {o.observation}\n"
            f"  Key patterns: {', '.join(o.key_patterns)}\n"
            f"  Raw: {o.raw_data_snippet}"
            for o in self.all_observations[-30:]  # Limit context size
        )

        existing_text = "\n".join(
            f"- [{p.confidence}] {p.pattern} (source: {p.source})"
            for p in self.patterns
        )

        prompt = prompt_template.format(
            task_context=task_context,
            observations=obs_text,
            existing_patterns=existing_text or "None yet",
        )

        result = self.llm.chat_json(prompt)
        if not result:
            return None

        # Handle both formats: direct array or object with "patterns" key
        if isinstance(result, list):
            pattern_list = result
            new_found = True
        elif isinstance(result, dict):
            pattern_list = result.get("patterns", [])
            new_found = result.get("new_patterns_found", True)
        else:
            return None

        self.patterns = []
        for p in pattern_list:
            self.patterns.append(MicroPattern(
                pattern=p.get("pattern", ""),
                evidence=p.get("evidence", []),
                confidence=p.get("confidence", "medium"),
                source=p.get("source", "single_class"),
                evidence_snippets=p.get("evidence_snippets", []),
            ))

        return {"new_patterns_found": new_found, "patterns": self.patterns}
