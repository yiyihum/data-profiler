from __future__ import annotations

import json
import textwrap
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.sandbox import LocalRestrictedRunner, execute_code


DATA_COLUMN_CANDIDATES = [
    "data",
    "comment",
    "text",
    "content",
    "message",
    "path",
    "filepath",
    "file",
]

LABEL_COLUMN_CANDIDATES = [
    "label",
    "target",
    "y",
    "class",
    "insult",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass(slots=True)
class IngestionResult:
    parquet_path: Path
    dataset_shape: tuple[int, int]
    task_type: str
    metadata: dict[str, Any]


class IngestionAgent:
    def __init__(
        self,
        sandbox_runner: LocalRestrictedRunner,
        default_task_description: str = "analyze the dataset",
        task_description_max_chars: int = 20_000,
        task_summary_max_chars: int = 2_000,
    ) -> None:
        self.sandbox_runner = sandbox_runner
        self.default_task_description = default_task_description.strip() or "analyze the dataset"
        self.task_description_max_chars = max(128, int(task_description_max_chars))
        self.task_summary_max_chars = max(64, int(task_summary_max_chars))

    def run(self, raw_path: str | Path, output_dir: str | Path) -> IngestionResult:
        source = Path(raw_path).resolve()
        output_root = Path(output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        source_type, resolved_source = self._detect_source(source)
        task_description, task_description_source, task_description_path = self._load_task_description(source, resolved_source)
        parquet_path = output_root / "standardized.parquet"

        payload = {
            "source_type": source_type,
            "source_path": str(resolved_source),
            "output_parquet_path": str(parquet_path),
        }
        loader_result = execute_code(self.sandbox_runner, self._build_loader_code(), payload)
        if not loader_result.ok:
            raise RuntimeError(f"Ingestion loader failed: {loader_result.error}")

        if not parquet_path.exists():
            raise RuntimeError(f"Ingestion loader did not produce parquet at {parquet_path}")

        standardized = pd.read_parquet(parquet_path)
        self._validate_standardized_schema(standardized)

        metadata: dict[str, Any] = {}
        if isinstance(loader_result.payload, dict):
            raw_metadata = loader_result.payload.get("metadata", {})
            if isinstance(raw_metadata, dict):
                metadata = raw_metadata

        metadata.update(
            {
                "source_type": source_type,
                "source_path": str(resolved_source),
                "dataset_shape": tuple(standardized.shape),
            }
        )
        metadata["task_description"] = task_description
        metadata["task_summary"] = task_description[: self.task_summary_max_chars]
        metadata["task_description_source"] = task_description_source
        metadata["task_description_path"] = task_description_path

        return IngestionResult(
            parquet_path=parquet_path,
            dataset_shape=tuple(standardized.shape),
            task_type="binary_classification",
            metadata=metadata,
        )

    def _load_task_description(self, raw_source: Path, resolved_source: Path) -> tuple[str, str, str | None]:
        candidates: list[Path] = []
        if raw_source.is_dir():
            candidates.append(raw_source / "description.md")
        if raw_source.is_file():
            candidates.append(raw_source.parent / "description.md")
        if resolved_source.is_dir():
            candidates.append(resolved_source / "description.md")
        if resolved_source.is_file():
            candidates.append(resolved_source.parent / "description.md")

        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)

            if not candidate.exists() or not candidate.is_file():
                continue

            with suppress(UnicodeDecodeError):
                text = candidate.read_text(encoding="utf-8").strip()
                if text:
                    return text[: self.task_description_max_chars], "file", str(candidate)

            with suppress(Exception):
                text = candidate.read_text(encoding="latin-1").strip()
                if text:
                    return text[: self.task_description_max_chars], "file", str(candidate)

        return self.default_task_description[: self.task_description_max_chars], "default", None

    def _detect_source(self, path: Path) -> tuple[str, Path]:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_file():
            ext = path.suffix.lower()
            if ext == ".csv":
                return "csv", path
            if ext == ".json":
                return "json", path
            if ext == ".parquet":
                return "parquet", path
            raise ValueError(f"Unsupported file extension: {ext}")

        train_csv = path / "train.csv"
        if train_csv.exists():
            return "csv", train_csv

        tabular_files = [
            p
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix.lower() in {".csv", ".json", ".parquet"}
        ]
        if tabular_files:
            ext = tabular_files[0].suffix.lower()
            if ext == ".csv":
                return "csv", tabular_files[0]
            if ext == ".json":
                return "json", tabular_files[0]
            return "parquet", tabular_files[0]

        image_files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if image_files:
            return "image_dir", path

        raise ValueError(f"Could not detect supported source type from directory: {path}")

    def _validate_standardized_schema(self, frame: pd.DataFrame) -> None:
        expected = {"data", "label"}
        missing = expected - set(frame.columns)
        if missing:
            raise ValueError(f"Standardized parquet missing columns: {sorted(missing)}")

        if frame.empty:
            raise ValueError("Standardized parquet is empty")

        if not pd.api.types.is_object_dtype(frame["data"]) and not pd.api.types.is_string_dtype(frame["data"]):
            raise ValueError("Column 'data' must be string-like")

    def _build_loader_code(self) -> str:
        data_columns = json.dumps(DATA_COLUMN_CANDIDATES, ensure_ascii=False)
        label_columns = json.dumps(LABEL_COLUMN_CANDIDATES, ensure_ascii=False)
        image_exts = json.dumps(sorted(IMAGE_EXTS), ensure_ascii=False)

        return textwrap.dedent(
            f"""
            from pathlib import Path

            import pandas as pd

            DATA_COLUMN_CANDIDATES = {data_columns}
            LABEL_COLUMN_CANDIDATES = {label_columns}
            IMAGE_EXTS = set({image_exts})


            def _load_source(source_type: str, source_path: str) -> pd.DataFrame:
                path = Path(source_path)
                if source_type == "csv":
                    return pd.read_csv(path)
                if source_type == "json":
                    return pd.read_json(path)
                if source_type == "parquet":
                    return pd.read_parquet(path)
                if source_type == "image_dir":
                    rows = [{{"data": str(p)}} for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
                    return pd.DataFrame(rows)
                raise ValueError(f"Unsupported source type: {{source_type}}")


            def _standardize(frame: pd.DataFrame, source_type: str, source_path: str):
                if frame.empty:
                    raise ValueError("Loaded dataset is empty")

                lower_map = {{col.lower(): col for col in frame.columns}}

                label_col = None
                for candidate in LABEL_COLUMN_CANDIDATES:
                    if candidate in lower_map:
                        label_col = lower_map[candidate]
                        break

                data_col = None
                for candidate in DATA_COLUMN_CANDIDATES:
                    if candidate in lower_map:
                        data_col = lower_map[candidate]
                        break

                if data_col is None:
                    candidates = [c for c in frame.columns if c != label_col]
                    if not candidates:
                        raise ValueError("Could not infer data column")
                    data_col = candidates[0]

                standardized = pd.DataFrame()
                standardized["data"] = frame[data_col].astype(str)

                if label_col is not None:
                    standardized["label"] = frame[label_col]
                else:
                    standardized["label"] = pd.Series([None] * len(standardized), dtype="object")

                metadata = {{
                    "data_column": data_col,
                    "label_column": label_col,
                    "null_data_rows": int(standardized["data"].isna().sum()),
                    "null_label_rows": int(standardized["label"].isna().sum()),
                    "source_type": source_type,
                    "source_path": source_path,
                }}

                if label_col is not None:
                    label_counts = standardized["label"].value_counts(dropna=False).to_dict()
                    metadata["label_distribution"] = {{str(k): int(v) for k, v in label_counts.items()}}

                return standardized, metadata


            def main(payload):
                source_type = payload["source_type"]
                source_path = payload["source_path"]
                output_path = Path(payload["output_parquet_path"])

                frame = _load_source(source_type, source_path)
                standardized, metadata = _standardize(frame, source_type, source_path)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                standardized.to_parquet(output_path, index=False)

                metadata["dataset_shape"] = [int(standardized.shape[0]), int(standardized.shape[1])]
                return {{"metadata": metadata}}
            """
        ).strip()
