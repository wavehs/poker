"""
OCR Preprocessing — Image preparation for poker text recognition.

Specialized preprocessing for different text types found on poker tables:
- Numeric text: pot, stack, bet amounts (high contrast, fixed font)
- Player names: alphanumeric, variable fonts
"""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_for_ocr(
    crop: np.ndarray,
    field_type: str = "generic",
    target_height: int = 64,
) -> np.ndarray:
    """
    Preprocess a cropped image region for OCR.

    Applies field-type-specific preprocessing pipeline.

    Args:
        crop: BGR image crop.
        field_type: Semantic type ('pot', 'stack', 'bet', 'player_name', 'generic').
        target_height: Target height for resize (preserves aspect ratio).

    Returns:
        Preprocessed grayscale image optimized for OCR.
    """
    if crop is None or crop.size == 0:
        return np.zeros((target_height, target_height), dtype=np.uint8)

    if field_type in ("pot", "stack", "bet", "blind"):
        return _preprocess_numeric(crop, target_height)
    elif field_type == "player_name":
        return _preprocess_text(crop, target_height)
    else:
        return _preprocess_generic(crop, target_height)


def _preprocess_numeric(crop: np.ndarray, target_height: int) -> np.ndarray:
    """
    Preprocessing for numeric text (pot, stack, bet amounts).

    Poker clients typically use high-contrast monospace fonts for numbers.
    Pipeline: grayscale → denoise → threshold → resize → pad.
    """
    gray = _to_grayscale(crop)
    gray = _denoise(gray, strength=5)

    # Invert if text is light on dark background
    gray = _auto_invert(gray)

    # Aggressive binarization for clean digit edges
    gray = _threshold_otsu(gray)

    # Resize maintaining aspect ratio
    gray = _resize_height(gray, target_height)

    # Add padding for OCR engine margins
    gray = _pad(gray, px=4)

    return gray


def _preprocess_text(crop: np.ndarray, target_height: int) -> np.ndarray:
    """
    Preprocessing for general text (player names).

    Less aggressive — preserve more detail for mixed fonts.
    """
    gray = _to_grayscale(crop)
    gray = _denoise(gray, strength=3)
    gray = _auto_invert(gray)

    # Adaptive threshold preserves more detail
    gray = _threshold_adaptive(gray)

    gray = _resize_height(gray, target_height)
    gray = _pad(gray, px=4)

    return gray


def _preprocess_generic(crop: np.ndarray, target_height: int) -> np.ndarray:
    """Generic preprocessing — moderate processing."""
    gray = _to_grayscale(crop)
    gray = _denoise(gray, strength=3)
    gray = _auto_invert(gray)
    gray = _resize_height(gray, target_height)
    gray = _pad(gray, px=2)
    return gray


def preprocess_fallback(crop: np.ndarray, target_height: int = 64) -> np.ndarray:
    """
    Fallback preprocessing for low-confidence OCR results.
    Applies contrast boost, 2x upscale, and thresholding.
    """
    if crop is None or crop.size == 0:
        return np.zeros((target_height, target_height), dtype=np.uint8)

    gray = _to_grayscale(crop)
    gray = _auto_invert(gray)

    # Contrast boost
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)

    # Upscale x2
    h, w = gray.shape[:2]
    gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Thresholding
    gray = _threshold_otsu(gray)

    # Resize to target height and pad
    gray = _resize_height(gray, target_height)
    gray = _pad(gray, px=4)

    return gray


# ─── Building blocks ────────────────────────────────────────────────────────


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale if needed."""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return img


def _denoise(gray: np.ndarray, strength: int = 5) -> np.ndarray:
    """Non-local means denoising."""
    if gray.size == 0:
        return gray
    return cv2.fastNlMeansDenoising(gray, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def _auto_invert(gray: np.ndarray) -> np.ndarray:
    """Invert if background is darker than text (most poker clients)."""
    if gray.size == 0:
        return gray
    mean_val = np.mean(gray)
    if mean_val < 128:
        return cv2.bitwise_not(gray)
    return gray


def _threshold_otsu(gray: np.ndarray) -> np.ndarray:
    """Otsu binarization for clean digit extraction."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _threshold_adaptive(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold for text with varying illumination."""
    block_size = max(11, (gray.shape[0] // 4) | 1)  # odd number >= 11
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 4
    )


def _resize_height(gray: np.ndarray, target_height: int) -> np.ndarray:
    """Resize to target height, preserving aspect ratio."""
    if gray.shape[0] == 0 or gray.shape[1] == 0:
        return gray
    h, w = gray.shape[:2]
    if h == target_height:
        return gray
    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_LINEAR)


def _pad(gray: np.ndarray, px: int = 4) -> np.ndarray:
    """Add white padding around the image."""
    return cv2.copyMakeBorder(
        gray, px, px, px, px, cv2.BORDER_CONSTANT, value=255
    )


def contrast_boost(img: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
    """
    Increase image contrast using cv2.convertScaleAbs.

    Args:
        img: Input image (BGR or Grayscale).
        alpha: Contrast control (1.0-3.0).
        beta: Brightness control (0-100).

    Returns:
        Contrast-boosted image.
    """
    if img is None or img.size == 0:
        return img
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def upscale_x2(img: np.ndarray) -> np.ndarray:
    """
    Upscale image by 2x using bicubic interpolation and apply thresholding.

    Args:
        img: Input image.

    Returns:
        Upscaled and thresholded image.
    """
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Apply thresholding
    gray = _to_grayscale(upscaled)
    gray = _auto_invert(gray)
    binary = _threshold_otsu(gray)

    # If the original image was BGR, we can optionally keep it binary grayscale
    # Since OCR backend accepts both, we'll return the binary image.
    return binary


# ─── Crop utilities ──────────────────────────────────────────────────────────


def crop_bbox(
    frame: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    padding: float = 0.05,
) -> np.ndarray:
    """
    Crop a bounding box region from frame with optional padding.

    Args:
        frame: Full BGR frame.
        x, y, w, h: Bounding box coordinates (pixels).
        padding: Fractional padding to add around the box.

    Returns:
        Cropped BGR region.
    """
    fh, fw = frame.shape[:2]
    pad_w = w * padding
    pad_h = h * padding

    x1 = max(0, int(x - pad_w))
    y1 = max(0, int(y - pad_h))
    x2 = min(fw, int(x + w + pad_w))
    y2 = min(fh, int(y + h + pad_h))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    return frame[y1:y2, x1:x2].copy()
