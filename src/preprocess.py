"""OpenCV image preprocessing: loading, deskewing, denoising, contrast enhancement
and image-quality features."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(src) -> np.ndarray:
    """Load an image (path, bytes or PIL/ndarray) as a BGR uint8 array.
    Uses imdecode so non-ASCII Windows paths work."""
    if isinstance(src, np.ndarray):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
    elif hasattr(src, "convert"):  # PIL image
        img = cv2.cvtColor(np.asarray(src.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imdecode(np.fromfile(str(Path(src)), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {src!r}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def estimate_skew(gray: np.ndarray, max_angle: float = 6.0) -> float:
    """Projection-profile skew estimation: the rotation that makes text rows
    most sharply separated maximises the variance of the row sums."""
    scale = 500.0 / max(gray.shape[1], 1)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # if the "ink" covers most of the image, polarity is reversed (light text on dark)
    if binary.mean() > 127:
        binary = 255 - binary

    def score(a: float) -> float:
        return float(np.var(_rotate(binary, a).sum(axis=1, dtype=np.float64)))

    coarse = np.arange(-max_angle, max_angle + 1e-6, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.5 + 1e-6, 0.1)
    return float(max(fine, key=score))


def preprocess(bgr: np.ndarray, min_width: int = 800, max_width: int = 1600) -> dict:
    """Full preprocessing chain. Returns every intermediate step (for the UI)
    plus the image fed to OCR and the scale factor applied."""
    h, w = bgr.shape[:2]
    scale = 1.0
    if w < min_width:
        scale = min_width / w
    elif w > max_width:
        scale = max_width / w
    resized = cv2.resize(bgr, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA) if scale != 1 else bgr

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew(gray)
    if abs(angle) >= 0.2:
        gray = _rotate(gray, angle)
        resized = _rotate(resized, angle)
    denoised = cv2.bilateralFilter(gray, 5, 40, 40)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)

    return dict(
        original=bgr, deskewed=resized, gray=gray, enhanced=enhanced, binary=binary,
        # The deep-learning OCR detector works best on the deskewed colour image;
        # CLAHE/binary outputs are kept for visualisation and quality features.
        ocr_input=resized, scale=scale, skew_angle=angle,
    )


def image_quality_features(bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 80, 160)
    noise = np.abs(gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)).mean()
    h, w = gray.shape
    return dict(
        img_sharpness=float(np.log1p(lap_var)),
        img_contrast=float(gray.std() / 255.0),
        img_brightness=float(gray.mean() / 255.0),
        img_edge_density=float((edges > 0).mean()),
        img_noise=float(noise),
        img_aspect=float(h / w),
    )
