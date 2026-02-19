from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


_COMMON_TEXT_COLUMNS = ("text", "comment", "data", "content", "message", "body")
_NON_TEXT_COLUMNS = {"date", "timestamp", "time", "id"}


@dataclass(slots=True)
class DataState:
    """Global mutable state shared by all agents."""

    df: pd.DataFrame
    target_col: str | None = None
    text_col: str | None = None
    coverage_mask: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.df.empty:
            raise ValueError("Input dataframe is empty")

        self.df = self.df.reset_index(drop=True)

        if self.target_col and self.target_col not in self.df.columns:
            raise ValueError(f"target_col not found: {self.target_col}")

        self.text_col = self.text_col or self._infer_text_col()
        if self.text_col not in self.df.columns:
            raise ValueError(f"text_col not found: {self.text_col}")

        self.coverage_mask = np.zeros(len(self.df), dtype=bool)

    @property
    def total_rows(self) -> int:
        return len(self.df)

    @property
    def coverage_ratio(self) -> float:
        return float(self.coverage_mask.sum() / max(self.total_rows, 1))

    @property
    def uncovered_ratio(self) -> float:
        return 1.0 - self.coverage_ratio

    def update_coverage(self, rule_mask: np.ndarray) -> None:
        mask = np.asarray(rule_mask, dtype=bool)
        if mask.shape != self.coverage_mask.shape:
            raise ValueError(
                f"Mask length mismatch: expected {self.coverage_mask.shape[0]}, got {mask.shape[0]}"
            )
        self.coverage_mask = np.logical_or(self.coverage_mask, mask)

    def get_uncovered_data(self) -> pd.DataFrame:
        return self.df.loc[~self.coverage_mask]

    def get_text_values(self) -> list[str]:
        return self.df[self.text_col].fillna("").astype(str).tolist()

    def _infer_text_col(self) -> str:
        lower_map = {col.lower(): col for col in self.df.columns}

        for col in _COMMON_TEXT_COLUMNS:
            mapped = lower_map.get(col)
            if mapped and mapped != self.target_col:
                return mapped

        for col in self.df.columns:
            if col == self.target_col:
                continue
            if pd.api.types.is_string_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
                if col.lower() in _NON_TEXT_COLUMNS:
                    continue
                return col

        for col in self.df.columns:
            if col != self.target_col:
                return col

        raise ValueError("Failed to infer a text column")
