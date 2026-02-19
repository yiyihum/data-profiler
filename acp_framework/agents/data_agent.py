from __future__ import annotations

import pandas as pd

from acp_framework.core.data_state import DataState


class DataAgent:
    """Curates dataset views using coverage-aware sampling strategies."""

    def __init__(self, data_state: DataState) -> None:
        self.state = data_state

    def get_view(self, strategy: str, sample_size: int = 40, random_state: int = 42) -> pd.DataFrame:
        if sample_size <= 0:
            return self.state.df.head(0)

        strategy = (strategy or "random_baseline").strip().lower()

        if strategy == "random_baseline":
            return self._sample(self.state.df, sample_size, random_state)

        if strategy == "uncovered_only":
            uncovered = self.state.get_uncovered_data()
            if uncovered.empty:
                return self.state.df.head(0)
            return self._sample(uncovered, sample_size, random_state)

        if strategy == "extreme_lengths":
            return self._extreme_lengths_view(sample_size)

        return self._sample(self.state.df, sample_size, random_state)

    def _extreme_lengths_view(self, sample_size: int) -> pd.DataFrame:
        source = self.state.get_uncovered_data()
        if source.empty:
            source = self.state.df

        text = source[self.state.text_col].fillna("").astype(str)
        lengths = text.str.len()
        half = max(sample_size // 2, 1)

        shortest = source.loc[lengths.nsmallest(half).index]
        longest = source.loc[lengths.nlargest(half).index]
        merged = pd.concat([shortest, longest], axis=0)
        merged = merged.loc[~merged.index.duplicated(keep="first")]

        if len(merged) > sample_size:
            merged = merged.iloc[:sample_size]
        return merged

    @staticmethod
    def _sample(frame: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
        if frame.empty:
            return frame
        n = min(sample_size, len(frame))
        return frame.sample(n=n, random_state=random_state)
