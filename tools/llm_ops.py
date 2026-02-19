from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from tools import cv_ops


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def _looks_like_image_path(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return Path(text).suffix in _IMAGE_EXTS


def _fallback_text_embedding(text: str) -> np.ndarray:
    counts = np.zeros(8, dtype=np.float32)
    if not text:
        return counts

    counts[0] = len(text)
    counts[1] = sum(ch.isalpha() for ch in text)
    counts[2] = sum(ch.isdigit() for ch in text)
    counts[3] = sum(ch.isspace() for ch in text)
    counts[4] = sum(ch in "!?" for ch in text)
    counts[5] = sum(ch in ",.;:" for ch in text)
    counts[6] = sum(ord(ch) > 127 for ch in text)
    counts[7] = float(hash(text) % 10_000) / 10_000.0
    return counts


def _image_embedding(path_str: str) -> np.ndarray:
    return np.array(
        [
            cv_ops.laplacian_variance(path_str),
            cv_ops.brightness_mean(path_str),
            float(len(path_str)),
        ],
        dtype=np.float32,
    )


def get_embeddings(values: list[Any], backend: str = "auto") -> np.ndarray:
    if not values:
        return np.zeros((0, 8), dtype=np.float32)

    as_strings = ["" if v is None else str(v) for v in values]
    image_like_ratio = sum(_looks_like_image_path(v) for v in values) / max(len(values), 1)

    if image_like_ratio > 0.8:
        return np.vstack([_image_embedding(v) for v in as_strings])

    if backend in {"auto", "hashing"}:
        try:
            from sklearn.feature_extraction.text import HashingVectorizer

            vectorizer = HashingVectorizer(n_features=128, alternate_sign=False, norm="l2")
            matrix = vectorizer.transform(as_strings)
            return matrix.toarray().astype(np.float32)
        except Exception:
            pass

    if backend in {"auto", "sentence_transformers"}:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            return np.asarray(model.encode(as_strings), dtype=np.float32)
        except Exception:
            pass

    return np.vstack([_fallback_text_embedding(text) for text in as_strings])


def generate_candidates_via_llm(
    samples: list[dict[str, Any]],
    context: dict[str, Any],
    n_candidates: int,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key_env: str = "OPENAI_API_KEY",
    temperature: float = 0.2,
    timeout_s: int = 30,
) -> list[dict[str, str]]:
    normalized_provider = provider.strip().lower()
    if normalized_provider != "openai":
        raise ValueError(f"Unsupported LLM provider: {provider}")

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing OpenAI API key environment variable: {api_key_env}")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = build_coder_prompt(samples, context, n_candidates)

    response = client.with_options(timeout=timeout_s).chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content or ""
    parsed = _parse_candidate_payload(raw)
    return parsed[:n_candidates]


def build_coder_prompt(samples: list[dict[str, Any]], context: dict[str, Any], n_candidates: int) -> str:
    task_description = str(context.get("task_description", "analyze the dataset")).strip() or "analyze the dataset"
    knowledge = str(context.get("knowledge", "")).strip()

    sample_lines: list[str] = []
    for idx, item in enumerate(samples, start=1):
        text = "" if item.get("data") is None else str(item.get("data"))
        label = item.get("label", None)
        sample_lines.append(f"{idx}. label={label} data={text[:300]}")

    joined_samples = "\n".join(sample_lines[:20])
    return (
        "You are a Data Detective.\n"
        f"Task Description: {task_description}\n"
        f"Current Knowledge: {knowledge}\n"
        f"Samples:\n{joined_samples}\n"
        f"Generate {n_candidates} candidate rules as JSON list with fields description and code.\n"
        "Each code must define: def check(x): return bool\n"
    )


_SYSTEM_PROMPT = (
    "You are an expert data profiling agent. "
    "Return strict JSON only. "
    "Each candidate must include description and Python code. "
    "Code must define exactly one function: def check(x): -> bool. "
    "No markdown, no explanation."
)


def _parse_candidate_payload(raw: str) -> list[dict[str, str]]:
    candidates = _json_payload_candidates(raw)
    valid: list[dict[str, str]] = []
    for item in candidates:
        description = str(item.get("description", "")).strip()
        code = str(item.get("code", "")).strip()
        if not description or not code:
            continue
        if "def check(" not in code:
            continue
        valid.append({"description": description, "code": code})
    return valid


def _json_payload_candidates(raw: str) -> list[dict[str, Any]]:
    payloads: list[Any] = []

    direct = _try_json(raw)
    if direct is not None:
        payloads.append(direct)

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    for chunk in fenced:
        parsed = _try_json(chunk)
        if parsed is not None:
            payloads.append(parsed)

    array_match = re.search(r"\[[\s\S]*\]", raw)
    if array_match:
        parsed = _try_json(array_match.group(0))
        if parsed is not None:
            payloads.append(parsed)

    obj_match = re.search(r"\{[\s\S]*\}", raw)
    if obj_match:
        parsed = _try_json(obj_match.group(0))
        if parsed is not None:
            payloads.append(parsed)

    for payload in payloads:
        normalized = _normalize_payload(payload)
        if normalized:
            return normalized
    return []


def _try_json(text: str) -> Any | None:
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def _normalize_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("candidates"), list):
            return [x for x in payload["candidates"] if isinstance(x, dict)]
        if isinstance(payload.get("rules"), list):
            return [x for x in payload["rules"] if isinstance(x, dict)]
    return []
