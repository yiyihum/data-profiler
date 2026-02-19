from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd


class HypothesorAgent:
    """Generate natural-language feature hypotheses from a data view."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.2,
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

    def propose(
        self,
        view_df: pd.DataFrame,
        intent: str,
        max_hypotheses: int = 3,
        skills_context: str = "",
    ) -> list[str]:
        if view_df.empty:
            return []

        if self.mock_mode or not self.enabled:
            return self._mock_hypotheses(max_hypotheses)

        prompt = self._build_prompt(view_df, intent, max_hypotheses, skills_context)
        raw = self._chat(prompt)
        hypotheses = _parse_hypothesis_list(raw)

        unique: list[str] = []
        seen: set[str] = set()
        for hypothesis in hypotheses:
            clean = hypothesis.strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(clean)
            if len(unique) >= max_hypotheses:
                break

        return unique

    def _build_prompt(
        self,
        view_df: pd.DataFrame,
        intent: str,
        max_hypotheses: int,
        skills_context: str,
    ) -> str:
        text_col = _infer_text_col(view_df)
        label_col = _infer_label_col(view_df)
        sample_lines = []
        rows = view_df.head(20)
        for idx, (_, row) in enumerate(rows.iterrows(), start=1):
            text = _truncate(str(row.get(text_col, "")), 240)
            if label_col is not None:
                sample_lines.append(f"{idx}. label={row.get(label_col)} text={json.dumps(text, ensure_ascii=False)}")
            else:
                sample_lines.append(f"{idx}. text={json.dumps(text, ensure_ascii=False)}")

        skills_block = skills_context.strip() or "None"

        return (
            "You are a data scientist focused on interpretable text-rule discovery.\n"
            f"Intent: {intent}\n"
            f"Known reusable skills:\n{skills_block}\n"
            "Important runtime contract:\n"
            "- Future rule function signature is def check(x): -> bool\n"
            "- x is always one raw text string (not a dict, not a dataframe row)\n"
            "- Do NOT reference any column names such as Insult/Date/Comment\n"
            "Task:\n"
            f"- Return exactly {max_hypotheses} hypotheses as strict JSON array of strings.\n"
            "- Hypotheses must describe string patterns only (keywords, regex, punctuation, casing, length, etc.).\n"
            "- Do not include markdown, code fences, or explanations.\n"
            "Samples:\n"
            + "\n".join(sample_lines)
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
                        "You generate high-quality natural-language feature hypotheses for data profiling. "
                        "Output must be strict JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _mock_hypotheses(max_hypotheses: int) -> list[str]:
        base = [
            "Text contains profanity or offensive terms",
            "Text is mostly uppercase letters",
            "Text contains URL-like patterns",
            "Text is very short (single phrase)",
        ]
        return base[:max_hypotheses]


def _parse_hypothesis_list(raw: str) -> list[str]:
    payload = _try_parse_json(raw)
    if isinstance(payload, list):
        return [str(item) for item in payload]

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    for block in fenced:
        payload = _try_parse_json(block)
        if isinstance(payload, list):
            return [str(item) for item in payload]

    lines = [line.strip(" -\t") for line in raw.splitlines() if line.strip()]
    return lines


def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _infer_text_col(view_df: pd.DataFrame) -> str:
    lowered = {c.lower(): c for c in view_df.columns}
    for key in ("comment", "text", "data", "content", "message", "body"):
        if key in lowered:
            return lowered[key]

    for col in view_df.columns:
        if pd.api.types.is_object_dtype(view_df[col]) or pd.api.types.is_string_dtype(view_df[col]):
            if col.lower() not in {"date", "timestamp", "time"}:
                return col

    return str(view_df.columns[-1])


def _infer_label_col(view_df: pd.DataFrame) -> str | None:
    lowered = {c.lower(): c for c in view_df.columns}
    for key in ("label", "target", "class", "y", "insult"):
        if key in lowered:
            return lowered[key]
    return None
