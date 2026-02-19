from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.engine import EngineResult
from schema.profile_protocol import Directive, ProfileDocument, ProfileMeta, ProfileStrategy


class ProfileCompiler:
    def compile(self, engine_result: EngineResult, output_path: str | Path) -> dict[str, Any]:
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        rules = sorted(engine_result.selected_rules, key=lambda x: x.get("score", 0.0), reverse=True)
        scores = [float(r.get("score", 0.0)) for r in rules]
        avg_score = float(np.mean(scores)) if scores else 0.0

        strategy = ProfileStrategy(
            difficulty=self._difficulty(avg_score),
            recommended_model="LightGBM + Custom Features",
        )

        shape = engine_result.metadata.get("dataset_shape", (0, 0))
        ingestion_meta = engine_result.metadata.get("ingestion", {})
        if not isinstance(ingestion_meta, dict):
            ingestion_meta = {}

        source = ingestion_meta.get("task_description_source")
        if source not in {"file", "default"}:
            source = None

        excerpt = ingestion_meta.get("task_summary")
        if excerpt is not None:
            excerpt = str(excerpt)

        description_path = ingestion_meta.get("task_description_path")
        if description_path is not None:
            description_path = str(description_path)

        meta = ProfileMeta(
            task="binary_classification",
            dataset_shape=tuple(shape),
            description_source=source,
            description_excerpt=excerpt,
            description_path=description_path,
        )

        directives: list[Directive] = []
        for idx, rule in enumerate(rules, start=1):
            score = float(rule.get("score", 0.0))
            directive = Directive(
                id=f"rule_{idx:03d}",
                type=self._directive_type(rule),
                priority=self._priority(score),
                insight=str(rule.get("description", "")),
                code=str(rule.get("code", "")),
                action=self._action_text(rule),
                score=score,
            )
            directives.append(directive)

        document = ProfileDocument(meta=meta, strategy=strategy, directives=directives)
        payload = document.model_dump()
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def _difficulty(self, avg_score: float) -> str:
        if avg_score >= 0.2:
            return "Hard"
        if avg_score >= 0.08:
            return "Medium"
        return "Easy"

    def _priority(self, score: float) -> str:
        if score >= 0.15:
            return "CRITICAL"
        if score >= 0.05:
            return "MEDIUM"
        return "LOW"

    def _directive_type(self, rule: dict[str, Any]) -> str:
        text = str(rule.get("description", "")).lower()
        if "drop" in text or "noise" in text or "http" in text:
            return "DATA_FILTER"
        return "FEATURE_ENGINEERING"

    def _action_text(self, rule: dict[str, Any]) -> str:
        description = str(rule.get("description", "rule"))
        text = description.lower()
        if "drop" in text or "noise" in text or "http" in text:
            return "Drop rows where rule returns True"
        return f"Create feature column using rule: {description}"
