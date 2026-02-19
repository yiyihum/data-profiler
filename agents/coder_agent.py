from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools import llm_ops


@dataclass(slots=True)
class RuleCandidate:
    description: str
    code: str
    source: str = "llm"


class CoderAgent:
    def __init__(
        self,
        llm_enabled: bool = True,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        llm_api_key_env: str = "OPENAI_API_KEY",
        llm_temperature: float = 0.2,
        llm_timeout_s: int = 30,
    ) -> None:
        self.llm_enabled = llm_enabled
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_api_key_env = llm_api_key_env
        self.llm_temperature = llm_temperature
        self.llm_timeout_s = llm_timeout_s

    def generate_candidates(
        self,
        samples: list[dict[str, Any]],
        context: dict[str, Any],
        n_candidates: int = 5,
    ) -> list[RuleCandidate]:
        if not self.llm_enabled:
            return []

        candidates: list[RuleCandidate] = []

        llm_candidates = llm_ops.generate_candidates_via_llm(
            samples=samples,
            context=context,
            n_candidates=n_candidates,
            provider=self.llm_provider,
            model=self.llm_model,
            api_key_env=self.llm_api_key_env,
            temperature=self.llm_temperature,
            timeout_s=self.llm_timeout_s,
        )
        for item in llm_candidates:
            code = item.get("code", "")
            description = item.get("description", "LLM generated candidate")
            if code:
                candidates.append(RuleCandidate(description=description, code=code, source="llm"))

        unique: list[RuleCandidate] = []
        seen_codes: set[str] = set()
        for candidate in candidates:
            if candidate.code in seen_codes:
                continue
            seen_codes.add(candidate.code)
            unique.append(candidate)
            if len(unique) >= n_candidates:
                break

        return unique
