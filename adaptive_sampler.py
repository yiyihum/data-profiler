import logging
from typing import Dict, List, Optional, Set

import pandas as pd

from code_executor import CodeExecutor
from config import Config
from llm_client import LLMClient
from models import Hypothesis, MicroPattern

logger = logging.getLogger(__name__)


class AdaptiveSampler:
    def __init__(self, data: pd.DataFrame, target_col: str, config: Config,
                 llm_client: LLMClient, executor: CodeExecutor):
        self.data = data
        self.target_col = target_col
        self.config = config
        self.llm = llm_client
        self.executor = executor
        self.sampled_ids: Set[int] = set()
        self.iteration = 0
        self.sampling_history: List[Dict] = []

    def initial_sample(self) -> pd.DataFrame:
        """Round 0: stratified random sampling by target variable."""
        self.iteration = 1
        n_per_class = self.config.micro_sample_per_class
        groups = self.data.groupby(self.target_col)
        samples = []
        for label, group in groups:
            available = group[~group.index.isin(self.sampled_ids)]
            k = min(n_per_class, len(available))
            sampled = available.sample(n=k, random_state=42)
            samples.append(sampled)
            self.sampled_ids.update(sampled.index.tolist())

        result = pd.concat(samples)
        self.sampling_history.append({
            "round": self.iteration,
            "mode": "stratified_random",
            "description": "Initial stratified random sampling by target variable",
            "n_sampled": len(result),
        })
        logger.info(f"Initial sampling: {len(result)} samples from {len(groups)} classes")
        return result

    def adaptive_sample(self, patterns: List[MicroPattern],
                        hypotheses: List[Hypothesis]) -> pd.DataFrame:
        """Subsequent rounds: LLM decides sampling strategy based on findings."""
        self.iteration += 1
        n = self.config.adaptive_sample_size

        # Build context for LLM
        pattern_desc = "\n".join(
            f"- [{p.confidence}] {p.pattern} (source: {p.source})"
            for p in patterns
        )
        hypothesis_desc = "\n".join(
            f"- {h.hypothesis} → {h.conclusion or 'pending'}"
            for h in hypotheses[:10]
        )
        history_desc = "\n".join(
            f"- Round {h['round']}: {h['mode']} - {h['description']} ({h['n_sampled']} samples)"
            for h in self.sampling_history
        )
        columns_desc = ", ".join(self.data.columns.tolist())

        prompt = f"""Based on the following analysis progress, decide the next sampling strategy.

## Discovered Patterns
{pattern_desc or "None yet"}

## Hypothesis Status
{hypothesis_desc or "None yet"}

## Previous Sampling History
{history_desc}

## Available Columns
{columns_desc}

## Target Column: {self.target_col}
## Dataset size: {len(self.data)} rows, already sampled: {len(self.sampled_ids)}

## Instructions
Decide how to sample the next {n} data points to maximize insight. You can:
1. Use a built-in mode: "stratified_random", "filter", "subgroup"
2. Create a custom mode with your own name (e.g., "counterexample", "boundary", "anomaly")

Return JSON:
{{
  "sampling_mode": "mode_name",
  "mode_description": "why this sampling approach",
  "sampling_code": "Python code that takes a DataFrame `df` and returns a filtered DataFrame. Use only pandas operations. Example: df[df['col'] > threshold]",
  "reason": "what you hope to learn"
}}"""

        result = self.llm.chat_json(prompt, system="You are a data analysis expert. Return valid JSON only.")

        if not result or "sampling_code" not in result:
            # Fallback: random sample from unsampled data
            logger.warning("Adaptive sampling LLM call failed, falling back to random")
            available = self.data[~self.data.index.isin(self.sampled_ids)]
            sampled = available.sample(n=min(n, len(available)), random_state=42 + self.iteration)
            self.sampled_ids.update(sampled.index.tolist())
            self.sampling_history.append({
                "round": self.iteration,
                "mode": "random_fallback",
                "description": "Fallback random sampling",
                "n_sampled": len(sampled),
            })
            return sampled

        # Execute the sampling code
        sampling_code = result["sampling_code"]
        exec_code = f"""import pandas as pd
import json

df = pd.read_json(DATA_PATH)
sampled_ids = {list(self.sampled_ids)}
df = df[~df.index.isin(sampled_ids)]

try:
    filtered = {sampling_code}
    if len(filtered) == 0:
        filtered = df
    result_indices = filtered.sample(n=min({n}, len(filtered)), random_state={42 + self.iteration}).index.tolist()
except Exception as e:
    result_indices = df.sample(n=min({n}, len(df)), random_state={42 + self.iteration}).index.tolist()

print(json.dumps(result_indices))
"""
        exec_result = self.executor.execute(exec_code)

        if exec_result.success:
            try:
                import json
                indices = json.loads(exec_result.stdout.strip())
                sampled = self.data.loc[indices]
            except Exception:
                available = self.data[~self.data.index.isin(self.sampled_ids)]
                sampled = available.sample(n=min(n, len(available)), random_state=42 + self.iteration)
        else:
            logger.warning(f"Sampling code execution failed: {exec_result.stderr}")
            available = self.data[~self.data.index.isin(self.sampled_ids)]
            sampled = available.sample(n=min(n, len(available)), random_state=42 + self.iteration)

        self.sampled_ids.update(sampled.index.tolist())
        self.sampling_history.append({
            "round": self.iteration,
            "mode": result.get("sampling_mode", "custom"),
            "description": result.get("mode_description", ""),
            "reason": result.get("reason", ""),
            "n_sampled": len(sampled),
        })
        logger.info(f"Adaptive sampling round {self.iteration}: {result.get('sampling_mode')} → {len(sampled)} samples")
        return sampled

    def get_sampling_summary(self) -> dict:
        """Return sampling coverage statistics."""
        coverage_by_class = {}
        for label, group in self.data.groupby(self.target_col):
            sampled_in_class = len(self.sampled_ids & set(group.index.tolist()))
            coverage_by_class[str(label)] = {
                "sampled": sampled_in_class,
                "total": len(group),
                "rate": sampled_in_class / len(group) if len(group) > 0 else 0,
            }
        return {
            "total_sampled": len(self.sampled_ids),
            "total_data": len(self.data),
            "iterations": self.iteration,
            "coverage_by_class": coverage_by_class,
            "sampling_history": self.sampling_history,
        }
