from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_image_array(path: str | Path) -> np.ndarray | None:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None

    try:
        import cv2  # type: ignore

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return image.astype(np.float32)
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore

        with Image.open(file_path) as img:
            gray = img.convert("L")
            return np.array(gray, dtype=np.float32)
    except Exception:
        return None


def laplacian_variance(path: str | Path) -> float:
    arr = _load_image_array(path)
    if arr is None:
        return 0.0

    try:
        import cv2  # type: ignore

        return float(cv2.Laplacian(arr, cv2.CV_64F).var())
    except Exception:
        gy, gx = np.gradient(arr)
        return float((gx**2 + gy**2).mean())


def brightness_mean(path: str | Path) -> float:
    arr = _load_image_array(path)
    if arr is None:
        return 0.0
    return float(arr.mean())
