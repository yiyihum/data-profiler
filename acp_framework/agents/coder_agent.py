from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(slots=True)
class RuleCandidate:
    description: str
    code: str
    source: str = "llm"


class CoderAgent:
    """Translate hypotheses into executable Python rule code."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.1,
        timeout_s: int = 30,
        enabled: bool = True,
        mock_mode: bool = False,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = float(temperature)
        self.timeout_s = int(timeout_s)
        self.enabled = bool(enabled)
        self.mock_mode = bool(mock_mode)

    def generate_batch(self, hypotheses: Iterable[str], context: str, skills_context: str = "") -> list[RuleCandidate]:
        candidates: list[RuleCandidate] = []

        for hypothesis in hypotheses:
            candidate = self.generate(hypothesis, context=context, skills_context=skills_context)
            if candidate is None:
                continue
            candidates.append(candidate)

        unique: list[RuleCandidate] = []
        seen: set[str] = set()
        for item in candidates:
            key = item.code.strip()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)

        return unique

    def generate(self, hypothesis_text: str, context: str, skills_context: str = "") -> RuleCandidate | None:
        if not hypothesis_text.strip():
            return None

        if self.mock_mode or not self.enabled:
            code = _mock_rule_from_hypothesis(hypothesis_text)
            return RuleCandidate(description=hypothesis_text, code=code, source="mock")

        prompt = self._build_prompt(hypothesis_text, context, skills_context)
        raw = self._chat(prompt)
        code = _extract_code(raw)
        if not code or "def check(" not in code:
            return None
        if _looks_like_row_based_rule(code):
            return None

        return RuleCandidate(description=hypothesis_text, code=code, source="llm")

    def _build_prompt(self, hypothesis_text: str, context: str, skills_context: str) -> str:
        skills_block = skills_context.strip() or "None"
        return (
            "Convert the hypothesis into Python code.\n"
            f"Context: {context}\n"
            f"Hypothesis: {hypothesis_text}\n"
            f"Known reusable skills:\n{skills_block}\n"
            "Runtime contract:\n"
            "- Function signature must be: def check(x): -> bool\n"
            "- x is one raw text value (string-like), NOT a dict or row object\n"
            "- Start with: text = '' if x is None else str(x)\n"
            "Rules:\n"
            "1) Return strict JSON object with key 'code'.\n"
            "2) code must define exactly one function: def check(x): -> bool.\n"
            "3) No markdown, no explanation, no imports beyond re/math/string/statistics.\n"
            "4) Rule must be deterministic and safe on arbitrary text input.\n"
            "5) Never use x['...'], x.get(...), row['...'], or row.get(...).\n"
        )

    def _chat(self, prompt: str) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing OpenAI API key in environment variable: {self.api_key_env}")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.with_options(timeout=self.timeout_s).chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Python rule generator. "
                        "Output must be strict JSON with one key: code."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


def _extract_code(raw: str) -> str:
    payload = _try_json(raw)
    if isinstance(payload, dict) and isinstance(payload.get("code"), str):
        return payload["code"].strip()

    fenced = re.findall(r"```(?:python|json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    for block in fenced:
        parsed = _try_json(block)
        if isinstance(parsed, dict) and isinstance(parsed.get("code"), str):
            return parsed["code"].strip()
        if "def check(" in block:
            return block.strip()

    if "def check(" in raw:
        start = raw.find("def check(")
        return raw[start:].strip()

    return ""


def _try_json(text: str) -> Any | None:
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def _mock_rule_from_hypothesis(hypothesis_text: str) -> str:
    lower = hypothesis_text.lower()

    if "url" in lower or "http" in lower:
        return (
            "def check(x):\n"
            "    text = '' if x is None else str(x).lower()\n"
            "    return 'http://' in text or 'https://' in text or 'www.' in text\n"
        )

    if "uppercase" in lower:
        return (
            "def check(x):\n"
            "    text = '' if x is None else str(x)\n"
            "    letters = [ch for ch in text if ch.isalpha()]\n"
            "    if not letters:\n"
            "        return False\n"
            "    upper = sum(ch.isupper() for ch in letters)\n"
            "    return upper / len(letters) >= 0.8\n"
        )

    return (
        "def check(x):\n"
        "    text = '' if x is None else str(x).lower()\n"
        "    keywords = ['idiot', 'stupid', 'fuck', 'moron', 'dumb']\n"
        "    return any(word in text for word in keywords)\n"
    )


def _looks_like_row_based_rule(code: str) -> bool:
    lowered = code.lower()
    blocked_patterns = ("x[", "x.get(", "row[", "row.get(")
    return any(token in lowered for token in blocked_patterns)
