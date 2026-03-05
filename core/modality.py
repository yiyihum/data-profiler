"""Modality Detection — Identify data types (tabular, image, audio, mixed) in a directory."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# Extension mappings
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".dcm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TABULAR_EXTENSIONS = {".csv", ".parquet", ".parq", ".tsv"}
JSON_EXTENSIONS = {".json", ".jsonl", ".ndjson"}
NUMPY_EXTENSIONS = {".npy", ".npz"}
HDF5_EXTENSIONS = {".h5", ".hdf5", ".hdf"}
STRUCTURED_EXTENSIONS = TABULAR_EXTENSIONS | JSON_EXTENSIONS | NUMPY_EXTENSIONS | HDF5_EXTENSIONS


@dataclass
class DataModality:
    """Describes the data modalities present in a directory."""

    primary: str  # "tabular" | "image" | "audio" | "text" | "mixed"
    has_tabular: bool = False
    has_images: bool = False
    has_audio: bool = False
    has_json: bool = False
    tabular_files: List[Path] = field(default_factory=list)
    image_files: List[Path] = field(default_factory=list)
    audio_files: List[Path] = field(default_factory=list)
    json_files: List[Path] = field(default_factory=list)
    other_files: List[Path] = field(default_factory=list)
    dir_tree: str = ""
    total_file_count: int = 0
    file_counts_by_ext: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary": self.primary,
            "has_tabular": self.has_tabular,
            "has_images": self.has_images,
            "has_audio": self.has_audio,
            "has_json": self.has_json,
            "tabular_file_count": len(self.tabular_files),
            "image_file_count": len(self.image_files),
            "audio_file_count": len(self.audio_files),
            "json_file_count": len(self.json_files),
            "other_file_count": len(self.other_files),
            "total_file_count": self.total_file_count,
            "file_counts_by_ext": self.file_counts_by_ext,
        }


def detect_modality(data_dir: Path) -> DataModality:
    """
    Walk a data directory and classify files by modality.

    Returns a DataModality describing what's present.
    """
    data_dir = Path(data_dir)

    tabular_files: List[Path] = []
    image_files: List[Path] = []
    audio_files: List[Path] = []
    json_files: List[Path] = []
    other_files: List[Path] = []
    ext_counts: Dict[str, int] = {}
    total = 0

    for fpath in data_dir.rglob("*"):
        if not fpath.is_file():
            continue
        # Skip hidden files and description.md
        if fpath.name.startswith(".") or fpath.name == "description.md":
            continue

        total += 1
        ext = fpath.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

        if ext in TABULAR_EXTENSIONS:
            tabular_files.append(fpath)
        elif ext in JSON_EXTENSIONS:
            json_files.append(fpath)
        elif ext in IMAGE_EXTENSIONS:
            image_files.append(fpath)
        elif ext in AUDIO_EXTENSIONS:
            audio_files.append(fpath)
        else:
            other_files.append(fpath)

    has_tabular = len(tabular_files) > 0
    has_images = len(image_files) > 0
    has_audio = len(audio_files) > 0
    has_json = len(json_files) > 0

    # JSON files count as structured/tabular for modality determination
    structured_count = len(tabular_files) + len(json_files)

    # Determine primary modality
    modality_counts = []
    if has_images:
        modality_counts.append(("image", len(image_files)))
    if has_audio:
        modality_counts.append(("audio", len(audio_files)))
    if structured_count > 0:
        modality_counts.append(("tabular", structured_count))

    if not modality_counts:
        primary = "unknown"
    elif len(modality_counts) == 1:
        primary = modality_counts[0][0]
    else:
        # If multiple modalities, check if non-tabular files vastly outnumber tabular
        non_tabular_count = len(image_files) + len(audio_files)
        if non_tabular_count > 10 and structured_count > 0:
            # Mixed: tabular metadata + non-tabular primary data
            primary = "mixed"
        else:
            # Pick the one with most files
            modality_counts.sort(key=lambda x: -x[1])
            primary = modality_counts[0][0]

    # Build directory tree (max depth=3, max entries=50)
    dir_tree = _build_dir_tree(data_dir, max_depth=3, max_entries=50)

    return DataModality(
        primary=primary,
        has_tabular=has_tabular or has_json,
        has_images=has_images,
        has_audio=has_audio,
        has_json=has_json,
        tabular_files=sorted(tabular_files),
        image_files=sorted(image_files),
        audio_files=sorted(audio_files),
        json_files=sorted(json_files),
        other_files=sorted(other_files),
        dir_tree=dir_tree,
        total_file_count=total,
        file_counts_by_ext=ext_counts,
    )


def sample_files(files: List[Path], n: int = 5) -> List[Path]:
    """Deterministic sampling: pick n evenly-spaced files from a sorted list."""
    if not files:
        return []
    files = sorted(files)
    if len(files) <= n:
        return files
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


def get_file_basic_info(fpath: Path) -> Dict[str, Any]:
    """
    Get basic info about a file. For images: dimensions/mode. For audio: duration/sample_rate.
    """
    fpath = Path(fpath)
    info: Dict[str, Any] = {
        "path": str(fpath),
        "name": fpath.name,
        "size_bytes": fpath.stat().st_size,
        "extension": fpath.suffix.lower(),
    }

    ext = fpath.suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        try:
            from PIL import Image
            with Image.open(fpath) as img:
                info["width"] = img.width
                info["height"] = img.height
                info["mode"] = img.mode
                info["format"] = img.format
        except Exception as e:
            info["error"] = f"Cannot read image: {e}"

    elif ext in AUDIO_EXTENSIONS:
        try:
            import torchaudio
            metadata = torchaudio.info(str(fpath))
            info["sample_rate"] = metadata.sample_rate
            info["channels"] = metadata.num_channels
            info["duration_sec"] = round(metadata.num_frames / metadata.sample_rate, 2)
            info["num_frames"] = metadata.num_frames
        except Exception as e:
            info["error"] = f"Cannot read audio: {e}"

    return info


def sniff_file_structure(fpath: Path) -> Dict[str, Any]:
    """
    Non-invasive structural probe for non-standard file formats.

    Returns a dict with keys like 'format', 'shape', 'keys', 'sample', etc.
    """
    fpath = Path(fpath)
    ext = fpath.suffix.lower()
    info: Dict[str, Any] = {"path": str(fpath), "extension": ext}

    if ext in JSON_EXTENSIONS:
        try:
            raw = fpath.read_bytes()[:500].decode("utf-8", errors="replace")
            stripped = raw.lstrip()
            if stripped.startswith("["):
                info["format"] = "json_array"
            elif stripped.startswith("{"):
                info["format"] = "json_object"
            else:
                info["format"] = "json_lines"
            # Try to detect keys
            import json as _json
            try:
                if ext in (".jsonl", ".ndjson"):
                    first_line = fpath.open().readline()
                    obj = _json.loads(first_line)
                else:
                    with open(fpath) as f:
                        obj = _json.load(f)
                    if isinstance(obj, list) and obj:
                        obj = obj[0]
                if isinstance(obj, dict):
                    info["keys"] = list(obj.keys())[:20]
                    info["sample_key_types"] = {
                        k: type(v).__name__ for k, v in list(obj.items())[:10]
                    }
            except Exception:
                pass
            info["preview"] = raw[:200]
        except Exception as e:
            info["error"] = str(e)

    elif ext in NUMPY_EXTENSIONS:
        try:
            if ext == ".npy":
                arr = np.load(str(fpath), mmap_mode="r")
                info["shape"] = list(arr.shape)
                info["dtype"] = str(arr.dtype)
                info["format"] = "npy"
            else:
                npz = np.load(str(fpath))
                info["keys"] = list(npz.keys())
                info["format"] = "npz"
                for key in list(npz.keys())[:3]:
                    info[f"{key}_shape"] = list(npz[key].shape)
                    info[f"{key}_dtype"] = str(npz[key].dtype)
        except Exception as e:
            info["error"] = str(e)

    elif ext in HDF5_EXTENSIONS:
        try:
            import h5py
            with h5py.File(str(fpath), "r") as f:
                info["format"] = "hdf5"
                info["keys"] = list(f.keys())[:20]
                for key in list(f.keys())[:5]:
                    ds = f[key]
                    if hasattr(ds, "shape"):
                        info[f"{key}_shape"] = list(ds.shape)
                        info[f"{key}_dtype"] = str(ds.dtype)
        except Exception as e:
            info["error"] = str(e)

    return info


def _build_dir_tree(root: Path, max_depth: int = 3, max_entries: int = 50) -> str:
    """Build a truncated directory tree string."""
    lines: List[str] = []
    _count = [0]

    def _walk(path: Path, prefix: str, depth: int):
        if _count[0] >= max_entries or depth > max_depth:
            return

        try:
            entries = sorted(path.iterdir())
        except PermissionError:
            return

        dirs = [e for e in entries if e.is_dir() and not e.name.startswith(".")]
        files = [e for e in entries if e.is_file() and not e.name.startswith(".")]

        # Show files at this level (summarize if many)
        if len(files) > 5:
            # Group by extension
            ext_groups: Dict[str, int] = {}
            for f in files:
                ext = f.suffix.lower() or "(no ext)"
                ext_groups[ext] = ext_groups.get(ext, 0) + 1
            for ext, count in sorted(ext_groups.items(), key=lambda x: -x[1]):
                lines.append(f"{prefix}{count} {ext} files")
                _count[0] += 1
                if _count[0] >= max_entries:
                    return
        else:
            for f in files:
                size = f.stat().st_size
                if size > 1e9:
                    size_str = f"{size/1e9:.1f}GB"
                elif size > 1e6:
                    size_str = f"{size/1e6:.1f}MB"
                elif size > 1e3:
                    size_str = f"{size/1e3:.0f}KB"
                else:
                    size_str = f"{size}B"
                lines.append(f"{prefix}{f.name} ({size_str})")
                _count[0] += 1
                if _count[0] >= max_entries:
                    return

        # Recurse into directories
        for d in dirs:
            n_files = sum(1 for _ in d.rglob("*") if _.is_file())
            lines.append(f"{prefix}{d.name}/ ({n_files} files)")
            _count[0] += 1
            if _count[0] >= max_entries:
                return
            _walk(d, prefix + "  ", depth + 1)

    _walk(root, "", 0)

    if _count[0] >= max_entries:
        lines.append("... (truncated)")

    return "\n".join(lines)
