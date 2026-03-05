import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self._data: Optional[pd.DataFrame] = None
        self._description: Optional[str] = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = self._load_data()
        return self._data

    @property
    def description(self) -> str:
        if self._description is None:
            self._description = self._load_description()
        return self._description

    def _load_data(self) -> pd.DataFrame:
        path = self.config.data_path
        if path.endswith(".json"):
            with open(path) as f:
                raw = json.load(f)
            df = pd.DataFrame(raw)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def _load_description(self) -> str:
        path = self.config.description_path
        if not path or not Path(path).exists():
            return "No description provided."
        return Path(path).read_text()

    def detect_target_column(self) -> str:
        """Detect likely target column from common patterns."""
        # Check boolean columns first
        for col in self.data.columns:
            if self.data[col].dtype == bool:
                return col
        # Check common target name patterns
        candidates = ["target", "label", "class", "outcome", "received"]
        for col in self.data.columns:
            col_lower = col.lower()
            if any(c in col_lower for c in candidates):
                return col
        # Fallback: last binary column
        for col in reversed(self.data.columns):
            nunique = self.data[col].nunique()
            if nunique == 2:
                return col
        return self.data.columns[-1]

    def get_feature_types(self) -> Dict[str, str]:
        """Classify each column's type."""
        types = {}
        for col in self.data.columns:
            dtype = self.data[col].dtype
            if dtype == "object":
                avg_len = self.data[col].dropna().astype(str).str.len().mean()
                if avg_len > 50:
                    types[col] = "text"
                else:
                    types[col] = "categorical"
            elif "datetime" in str(dtype) or "timestamp" in col.lower():
                types[col] = "datetime"
            elif "bool" in str(dtype):
                types[col] = "categorical"
            else:
                nunique = self.data[col].nunique()
                if nunique <= 10:
                    types[col] = "categorical"
                else:
                    types[col] = "numeric"
        return types

    def get_preview(self) -> str:
        """Generate a text preview of the dataset for LLM consumption."""
        df = self.data
        lines = []
        lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        lines.append(f"\nColumn types:\n{df.dtypes.to_string()}")
        lines.append(f"\nMissing values:\n{df.isnull().sum().to_string()}")

        # Basic stats for numeric columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            lines.append(f"\nNumeric statistics:\n{df[numeric_cols].describe().to_string()}")

        # Categorical/text column summaries
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        for col in cat_cols[:5]:
            nunique = df[col].nunique()
            if nunique <= 20:
                lines.append(f"\n{col} value counts:\n{df[col].value_counts().head(10).to_string()}")
            else:
                avg_len = df[col].dropna().astype(str).str.len().mean()
                lines.append(f"\n{col}: {nunique} unique values, avg length: {avg_len:.0f} chars")

        # Target distribution (try to detect)
        target = self.detect_target_column()
        lines.append(f"\nTarget column (detected): {target}")
        lines.append(f"Target distribution:\n{df[target].value_counts().to_string()}")

        return "\n".join(lines)

    def sample_data(self, n: int, target_col: str, stratified: bool = True,
                    exclude_ids: set = None) -> pd.DataFrame:
        """Sample data, optionally stratified by target."""
        df = self.data
        if exclude_ids:
            df = df[~df.index.isin(exclude_ids)]
        if stratified and target_col in df.columns:
            groups = df.groupby(target_col)
            per_class = max(1, n // len(groups))
            samples = []
            for _, group in groups:
                k = min(per_class, len(group))
                samples.append(group.sample(n=k, random_state=42 + len(exclude_ids or set())))
            return pd.concat(samples).head(n)
        return df.sample(n=min(n, len(df)), random_state=42)

    def format_data_point(self, row: pd.Series) -> str:
        """Format a single data point for LLM analysis."""
        parts = []
        for col, val in row.items():
            if isinstance(val, list):
                val_str = str(val[:5]) + (f"... ({len(val)} items)" if len(val) > 5 else "")
            elif isinstance(val, str) and len(val) > 500:
                val_str = val[:500] + "..."
            else:
                val_str = str(val)
            parts.append(f"  {col}: {val_str}")
        return "\n".join(parts)
