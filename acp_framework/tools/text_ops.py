from __future__ import annotations

import re
from typing import Iterable


def safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def uppercase_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch.isupper() for ch in letters) / len(letters)


def has_url(text: str) -> bool:
    lower = text.lower()
    return "http://" in lower or "https://" in lower or "www." in lower
