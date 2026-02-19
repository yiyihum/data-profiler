from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SkillEntry:
    fingerprint: str
    description: str
    code: str
    intent: str
    score: float


class SkillsLibrary:
    """Simple persistent memory of high-value reusable rules."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path).resolve() if db_path else None
        self._skills: list[SkillEntry] = []
        if self.db_path and self.db_path.exists():
            self._load()

    def register(self, rule: Any) -> None:
        description = str(getattr(rule, "description", "")).strip()
        code = str(getattr(rule, "code", "")).strip()
        intent = str(getattr(rule, "intent", "")).strip() or "feature_mining"
        score = float(getattr(rule, "score", 0.0))

        if not code:
            return

        fingerprint = _fingerprint_code(code)
        if any(item.fingerprint == fingerprint for item in self._skills):
            return

        self._skills.append(
            SkillEntry(
                fingerprint=fingerprint,
                description=description,
                code=code,
                intent=intent,
                score=score,
            )
        )

    def shortlist(self, intent: str, max_items: int = 5) -> list[SkillEntry]:
        normalized_intent = intent.strip().lower()
        matched = [s for s in self._skills if s.intent.strip().lower() == normalized_intent]
        if not matched:
            matched = self._skills

        matched.sort(key=lambda item: item.score, reverse=True)
        return matched[:max_items]

    def render_prompt_context(self, intent: str, max_items: int = 5) -> str:
        items = self.shortlist(intent, max_items=max_items)
        if not items:
            return "None"

        lines = []
        for idx, item in enumerate(items, start=1):
            lines.append(f"[{idx}] {item.description} | score={item.score:.4f}")
            lines.append(item.code)
        return "\n".join(lines)

    def flush(self) -> None:
        if not self.db_path:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [asdict(item) for item in self._skills]
        self.db_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        raw = json.loads(self.db_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return

        loaded: list[SkillEntry] = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            try:
                loaded.append(
                    SkillEntry(
                        fingerprint=str(row.get("fingerprint", "")),
                        description=str(row.get("description", "")),
                        code=str(row.get("code", "")),
                        intent=str(row.get("intent", "feature_mining")),
                        score=float(row.get("score", 0.0)),
                    )
                )
            except Exception:
                continue

        self._skills = [item for item in loaded if item.fingerprint and item.code]


def _fingerprint_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
