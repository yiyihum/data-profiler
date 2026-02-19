from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProfileMeta(BaseModel):
    task: str = "binary_classification"
    dataset_shape: tuple[int, int]
    description_source: Literal["file", "default"] | None = None
    description_excerpt: str | None = None
    description_path: str | None = None


class ProfileStrategy(BaseModel):
    difficulty: Literal["Easy", "Medium", "Hard"]
    recommended_model: str


class Directive(BaseModel):
    id: str
    type: Literal["FEATURE_ENGINEERING", "DATA_FILTER"]
    priority: Literal["LOW", "MEDIUM", "CRITICAL"]
    insight: str
    code: str
    action: str
    score: float = Field(ge=0.0)


class ProfileDocument(BaseModel):
    meta: ProfileMeta
    strategy: ProfileStrategy
    directives: list[Directive]
