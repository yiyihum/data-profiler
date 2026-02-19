from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MetadataStore:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, item: dict[str, Any]) -> None:
        self.entries.append(item)

    def summary(self, limit: int = 8) -> str:
        if not self.entries:
            return "No discovered patterns yet."

        lines: list[str] = []
        for idx, entry in enumerate(self.entries[-limit:], start=1):
            rule = entry.get("description", "unknown rule")
            score = entry.get("score", 0.0)
            depth = entry.get("depth", "?")
            lines.append(f"{idx}. depth={depth}, score={score:.4f}, rule={rule}")
        return "\n".join(lines)
