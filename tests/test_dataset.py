"""Tests for Dataset Pipeline and Augmentations."""

import numpy as np
import pytest

from data.dataset import (
    Annotation,
    PokerFrameDataset,
    YOLO_CLASSES,
    YOLO_CLASS_NAMES,
    generate_synthetic_frame,
)
from data.augment import (
    apply_random_augmentations,
    augment_brightness_contrast,
    augment_color_shift,
    augment_noise,
)


# ─── Annotation ──────────────────────────────────────────────────────────────


class TestAnnotation:
    def test_to_yolo_line(self):
        ann = Annotation(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.2)
        line = ann.to_yolo_line()
        assert line.startswith("0 ")
        parts = line.split()
        assert len(parts) == 5

    def test_from_yolo_line(self):
        line = "0 0.500000 0.500000 0.100000 0.200000"
        ann = Annotation.from_yolo_line(line)
        assert ann.class_id == 0
        assert abs(ann.cx - 0.5) < 1e-4
        assert abs(ann.w - 0.1) < 1e-4

    def test_roundtrip(self):
        ann = Annotation(class_id=3, cx=0.123, cy=0.456, w=0.078, h=0.091)
        line = ann.to_yolo_line()
        ann2 = Annotation.from_yolo_line(line)
        assert ann2.class_id == ann.class_id
        assert abs(ann2.cx - ann.cx) < 1e-4
        assert abs(ann2.cy - ann.cy) < 1e-4


# ─── Synthetic frame generation ─────────────────────────────────────────────


class TestSyntheticFrame:
    def test_generate_default(self):
        frame, annotations = generate_synthetic_frame()
        assert frame.shape == (1080, 1920, 3)
        assert len(annotations) > 0

    def test_annotations_normalized(self):
        _, annotations = generate_synthetic_frame()
        for ann in annotations:
            assert 0 <= ann.cx <= 1
            assert 0 <= ann.cy <= 1
            assert 0 < ann.w <= 1
            assert 0 < ann.h <= 1
            assert ann.class_id in range(len(YOLO_CLASS_NAMES))

    def test_contains_expected_classes(self):
        _, annotations = generate_synthetic_frame(num_cards=7, num_players=6)
        class_ids = {a.class_id for a in annotations}
        assert YOLO_CLASSES["card"] in class_ids
        assert YOLO_CLASSES["pot"] in class_ids
        assert YOLO_CLASSES["dealer_button"] in class_ids
        assert YOLO_CLASSES["player_panel"] in class_ids


# ─── YOLO class mapping ─────────────────────────────────────────────────────


class TestYOLOClasses:
    def test_all_classes_mapped(self):
        assert len(YOLO_CLASSES) == 8

    def test_class_ids_sequential(self):
        ids = sorted(YOLO_CLASSES.values())
        assert ids == list(range(len(ids)))

    def test_class_names_match(self):
        assert len(YOLO_CLASS_NAMES) == len(YOLO_CLASSES)


# ─── PokerFrameDataset ──────────────────────────────────────────────────────


class TestPokerFrameDataset:
    def test_generate_synthetic(self, tmp_path):
        ds = PokerFrameDataset(tmp_path / "ds")
        n = ds.generate_synthetic(count=5)
        assert n == 5

        images = list((tmp_path / "ds" / "images").glob("*.png"))
        labels = list((tmp_path / "ds" / "labels").glob("*.txt"))
        assert len(images) == 5
        assert len(labels) == 5

    def test_validate_good(self, tmp_path):
        ds = PokerFrameDataset(tmp_path / "ds")
        ds.generate_synthetic(count=3)
        result = ds.validate()
        assert result["valid"]
        assert result["total_images"] == 3
        assert result["total_labels"] == 3

    def test_validate_missing_dirs(self, tmp_path):
        ds = PokerFrameDataset(tmp_path / "empty")
        result = ds.validate()
        assert not result["valid"]

    def test_split(self, tmp_path):
        ds = PokerFrameDataset(tmp_path / "ds")
        ds.generate_synthetic(count=10)
        counts = ds.split(train=0.7, val=0.2, test=0.1)

        assert counts["train"] == 7
        assert counts["val"] == 2
        assert counts["test"] == 1

        # Check files exist
        train_imgs = list((tmp_path / "ds" / "train" / "images").glob("*"))
        assert len(train_imgs) == 7

    def test_export_yolo_yaml(self, tmp_path):
        ds = PokerFrameDataset(tmp_path / "ds")
        yaml_path = ds.export_yolo_yaml()
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "nc: 8" in content
        assert "train:" in content
        assert "card" in content


# ─── Augmentations ───────────────────────────────────────────────────────────


class TestAugmentations:
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

    def test_brightness_contrast(self, sample_image):
        result = augment_brightness_contrast(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_noise(self, sample_image):
        result = augment_noise(sample_image)
        assert result.shape == sample_image.shape
        # Should be different from original
        assert not np.array_equal(result, sample_image)

    def test_color_shift(self, sample_image):
        result = augment_color_shift(sample_image)
        assert result.shape == sample_image.shape

    def test_apply_random(self, sample_image):
        result = apply_random_augmentations(sample_image, probability=1.0)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_pipeline_no_crash(self, sample_image):
        """Apply augmentations many times — should never crash."""
        for _ in range(20):
            result = apply_random_augmentations(sample_image, probability=0.8)
            assert result.shape == sample_image.shape
