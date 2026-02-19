from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from agents.coder_agent import CoderAgent
from agents.sampler_agent import SamplerAgent
from core.memory import MetadataStore
from core.sandbox import LocalRestrictedRunner, RuleEvalResult
from tools import llm_ops


@dataclass(slots=True)
class EngineConfig:
    max_depth: int = 3
    samples_per_node: int = 5
    candidates_per_node: int = 5
    purity_threshold: float = 0.98
    min_samples_split: int = 40
    timeout_s: int = 5
    embedding_backend: str = "auto"


@dataclass(slots=True)
class EngineResult:
    tree: dict[str, Any]
    selected_rules: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(slots=True)
class NodeContext:
    node_id: str
    depth: int
    indices: np.ndarray


class CDTEngine:
    def __init__(
        self,
        sampler: SamplerAgent,
        coder: CoderAgent,
        sandbox: LocalRestrictedRunner,
        memory: MetadataStore | None = None,
    ) -> None:
        self.sampler = sampler
        self.coder = coder
        self.sandbox = sandbox
        self.memory = memory or MetadataStore()
        self.selected_rules: list[dict[str, Any]] = []

    def run(self, parquet_path: str, config: EngineConfig, metadata: dict[str, Any] | None = None) -> EngineResult:
        frame = pd.read_parquet(parquet_path)
        if "data" not in frame.columns or "label" not in frame.columns:
            raise ValueError("Input parquet must contain columns: data, label")

        values = frame["data"].astype(str).tolist()
        labels = self._prepare_labels(frame["label"])
        embeddings = llm_ops.get_embeddings(values, backend=config.embedding_backend)

        self.selected_rules = []
        root_indices = np.arange(len(frame), dtype=int)
        root_ctx = NodeContext(node_id="root", depth=0, indices=root_indices)

        tree = self._process_node(
            ctx=root_ctx,
            values=values,
            labels=labels,
            embeddings=embeddings,
            config=config,
            run_metadata=metadata or {},
        )

        engine_metadata = {
            "dataset_shape": tuple(frame.shape),
            "has_labels": labels is not None,
            "rules_selected": len(self.selected_rules),
        }
        if metadata:
            engine_metadata.update({"ingestion": metadata})

        return EngineResult(tree=tree, selected_rules=self.selected_rules, metadata=engine_metadata)

    def _process_node(
        self,
        ctx: NodeContext,
        values: list[str],
        labels: np.ndarray | None,
        embeddings: np.ndarray,
        config: EngineConfig,
        run_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        node_values = [values[i] for i in ctx.indices]
        node_labels = labels[ctx.indices] if labels is not None else None
        node_emb = embeddings[ctx.indices]

        purity = self._purity(node_labels)
        node = {
            "node_id": ctx.node_id,
            "depth": ctx.depth,
            "size": int(ctx.indices.size),
            "purity": purity,
            "is_leaf": False,
            "stop_reason": None,
            "rule": None,
            "children": {},
        }

        stop_reason = self._stop_reason(ctx, node_labels, config)
        if stop_reason:
            node["is_leaf"] = True
            node["stop_reason"] = stop_reason
            return node

        local_indices = self.sampler.select_indices(
            embeddings=node_emb,
            labels=node_labels,
            k=min(config.samples_per_node, ctx.indices.size),
            strategy_hint=None,
        )

        sampled_global = [int(ctx.indices[i]) for i in local_indices]
        samples = [
            {
                "index": idx,
                "data": values[idx],
                "label": None if labels is None else str(labels[idx]),
            }
            for idx in sampled_global
        ]

        context = {
            "depth": ctx.depth,
            "node_size": int(ctx.indices.size),
            "knowledge": self.memory.summary(),
            "source_type": run_metadata.get("source_type", "unknown"),
            "task_description": run_metadata.get("task_description", "analyze the dataset"),
            "task_summary": run_metadata.get("task_summary", "analyze the dataset"),
            "task_description_source": run_metadata.get("task_description_source", "default"),
        }

        candidates = self.coder.generate_candidates(samples=samples, context=context, n_candidates=config.candidates_per_node)
        evals = self.sandbox.validate_rules(rule_candidates=candidates, data_values=node_values)

        best = self._select_best(evals, node_labels, node_emb)
        if best is None:
            node["is_leaf"] = True
            node["stop_reason"] = "no_valid_candidate"
            return node

        best_eval, best_score = best
        mask = best_eval.mask
        true_local = np.where(mask)[0]
        false_local = np.where(~mask)[0]

        if true_local.size == 0 or false_local.size == 0:
            node["is_leaf"] = True
            node["stop_reason"] = "degenerate_split"
            return node

        true_global = ctx.indices[true_local]
        false_global = ctx.indices[false_local]

        rule_id = f"rule_{len(self.selected_rules) + 1:03d}"
        rule_record = {
            "id": rule_id,
            "node_id": ctx.node_id,
            "depth": ctx.depth,
            "description": best_eval.candidate.description,
            "code": best_eval.candidate.code,
            "score": float(best_score),
            "runtime_ms": int(best_eval.runtime_ms),
            "support": int(mask.sum()),
            "size": int(mask.size),
        }
        self.selected_rules.append(rule_record)
        self.memory.add(rule_record)

        node["rule"] = {
            "id": rule_id,
            "description": best_eval.candidate.description,
            "score": float(best_score),
            "runtime_ms": int(best_eval.runtime_ms),
            "code": best_eval.candidate.code,
        }

        left_ctx = NodeContext(node_id=f"{ctx.node_id}.L", depth=ctx.depth + 1, indices=false_global)
        right_ctx = NodeContext(node_id=f"{ctx.node_id}.R", depth=ctx.depth + 1, indices=true_global)

        node["children"] = {
            "false": self._process_node(left_ctx, values, labels, embeddings, config, run_metadata),
            "true": self._process_node(right_ctx, values, labels, embeddings, config, run_metadata),
        }
        return node

    def _select_best(
        self,
        evals: list[RuleEvalResult],
        labels: np.ndarray | None,
        embeddings: np.ndarray,
    ) -> tuple[RuleEvalResult, float] | None:
        scored: list[tuple[RuleEvalResult, float]] = []
        for item in evals:
            if not item.valid:
                continue
            if item.mask.size == 0:
                continue
            score = self._score_split(item.mask, labels, embeddings)
            if np.isfinite(score) and score > 0:
                scored.append((item, float(score)))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0]

    def _score_split(self, mask: np.ndarray, labels: np.ndarray | None, embeddings: np.ndarray) -> float:
        if mask.size == 0:
            return float("-inf")
        if mask.sum() == 0 or mask.sum() == mask.size:
            return float("-inf")

        if labels is not None and np.unique(labels).size >= 2:
            return self._information_gain(labels, mask)

        return self._proxy_score(mask, embeddings)

    def _information_gain(self, labels: np.ndarray, mask: np.ndarray) -> float:
        parent_entropy = self._entropy(labels)

        left = labels[~mask]
        right = labels[mask]
        left_weight = left.size / labels.size
        right_weight = right.size / labels.size

        child_entropy = left_weight * self._entropy(left) + right_weight * self._entropy(right)
        return float(parent_entropy - child_entropy)

    def _proxy_score(self, mask: np.ndarray, embeddings: np.ndarray) -> float:
        n = mask.size
        pos = int(mask.sum())
        neg = n - pos
        balance = 1.0 - abs(pos - neg) / max(n, 1)

        left = embeddings[~mask]
        right = embeddings[mask]
        if left.size == 0 or right.size == 0:
            separation = 0.0
        else:
            left_center = left.mean(axis=0)
            right_center = right.mean(axis=0)
            separation = float(np.linalg.norm(left_center - right_center))
            separation = separation / (1.0 + separation)

        return 0.7 * balance + 0.3 * separation

    def _entropy(self, labels: np.ndarray) -> float:
        if labels.size == 0:
            return 0.0

        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-(probs * np.log2(probs)).sum())

    def _purity(self, labels: np.ndarray | None) -> float:
        if labels is None or labels.size == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        return float(counts.max() / counts.sum())

    def _stop_reason(self, ctx: NodeContext, labels: np.ndarray | None, config: EngineConfig) -> str | None:
        if ctx.depth >= config.max_depth:
            return "max_depth"
        if ctx.indices.size < config.min_samples_split:
            return "min_samples_split"
        purity = self._purity(labels)
        if purity >= config.purity_threshold and labels is not None:
            return "purity_threshold"
        return None

    def _prepare_labels(self, series: pd.Series) -> np.ndarray | None:
        non_null = series.dropna()
        if non_null.empty:
            return None

        unique = non_null.astype(str).unique()
        if unique.size <= 1:
            return None

        normalized = series.fillna("__MISSING__").astype(str).to_numpy()
        return normalized
