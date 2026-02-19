from __future__ import annotations

import re
from typing import Pattern


def safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def safe_len(value: object) -> int:
    return len(safe_str(value))


def starts_with(value: object, prefix: str, ignore_case: bool = True) -> bool:
    text = safe_str(value)
    if ignore_case:
        return text.lower().startswith(prefix.lower())
    return text.startswith(prefix)


def contains_regex(value: object, pattern: str | Pattern[str]) -> bool:
    text = safe_str(value)
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    return compiled.search(text) is not None


def has_keyword(value: object, keywords: list[str]) -> bool:
    text = safe_str(value).lower()
    return any(keyword.lower() in text for keyword in keywords)
