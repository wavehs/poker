"""
Data Augmentations — Poker-specific image augmentations for training.

Provides augmentation functions that preserve YOLO-format annotations.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


def augment_brightness_contrast(
    image: np.ndarray,
    alpha_range: tuple[float, float] = (0.7, 1.3),
    beta_range: tuple[int, int] = (-30, 30),
) -> np.ndarray:
    """
    Random brightness and contrast adjustment.

    Args:
        image: BGR image.
        alpha_range: Contrast multiplier range.
        beta_range: Brightness offset range.

    Returns:
        Augmented image.
    """
    alpha = random.uniform(*alpha_range)
    beta = random.randint(*beta_range)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def augment_noise(
    image: np.ndarray,
    sigma_range: tuple[float, float] = (5.0, 25.0),
) -> np.ndarray:
    """Add Gaussian noise."""
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def augment_color_shift(
    image: np.ndarray,
    hue_range: int = 15,
    sat_range: int = 30,
) -> np.ndarray:
    """Random hue and saturation shift (simulates different table themes)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)

    hue_shift = random.randint(-hue_range, hue_range)
    sat_shift = random.randint(-sat_range, sat_range)

    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_blur(
    image: np.ndarray,
    ksize_range: tuple[int, int] = (1, 3),
) -> np.ndarray:
    """Random Gaussian blur."""
    ksize = random.choice(range(ksize_range[0], ksize_range[1] + 1, 2))
    if ksize < 3:
        return image
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def augment_small_rotation(
    image: np.ndarray,
    max_angle: float = 3.0,
) -> np.ndarray:
    """
    Small rotation (simulates slight camera tilt).
    Note: does NOT modify annotations — for small angles, bbox error is negligible.
    """
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def augment_jpeg_compression(
    image: np.ndarray,
    quality_range: tuple[int, int] = (50, 90),
) -> np.ndarray:
    """Simulate JPEG compression artifacts."""
    quality = random.randint(*quality_range)
    _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# ─── Pipeline ────────────────────────────────────────────────────────────────


def apply_random_augmentations(
    image: np.ndarray,
    probability: float = 0.5,
) -> np.ndarray:
    """
    Apply a random subset of augmentations with given probability.

    Each augmentation is applied independently with `probability` chance.

    Args:
        image: BGR image.
        probability: Probability of applying each augmentation.

    Returns:
        Augmented image.
    """
    result = image.copy()

    if random.random() < probability:
        result = augment_brightness_contrast(result)

    if random.random() < probability:
        result = augment_color_shift(result)

    if random.random() < probability * 0.5:  # noise less often
        result = augment_noise(result)

    if random.random() < probability * 0.3:  # blur rarely
        result = augment_blur(result)

    if random.random() < probability * 0.3:  # rotation rarely
        result = augment_small_rotation(result)

    if random.random() < probability * 0.4:
        result = augment_jpeg_compression(result)

    return result
