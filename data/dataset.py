"""
Dataset Pipeline — Tools for creating, validating, and managing poker frame datasets.

Supports YOLO-format annotations for training object detection models.

Usage:
    python -m data.dataset generate --count 100 --output data/labeled_frames
    python -m data.dataset validate --path data/labeled_frames
    python -m data.dataset split --path data/labeled_frames --ratio 0.7 0.2 0.1
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from libs.common.schemas import DetectionClass

# ─── YOLO class mapping ─────────────────────────────────────────────────────

YOLO_CLASSES: dict[str, int] = {
    DetectionClass.CARD.value: 0,
    DetectionClass.CHIP_STACK.value: 1,
    DetectionClass.POT.value: 2,
    DetectionClass.DEALER_BUTTON.value: 3,
    DetectionClass.PLAYER_PANEL.value: 4,
    DetectionClass.BET_AMOUNT.value: 5,
    DetectionClass.ACTION_BUTTON.value: 6,
    DetectionClass.BOARD_AREA.value: 7,
}

YOLO_CLASS_NAMES: list[str] = [
    cls for cls, _ in sorted(YOLO_CLASSES.items(), key=lambda x: x[1])
]


# ─── Annotation ──────────────────────────────────────────────────────────────


class Annotation:
    """Single YOLO-format annotation (class_id, cx, cy, w, h — all normalized)."""

    __slots__ = ("class_id", "cx", "cy", "w", "h")

    def __init__(self, class_id: int, cx: float, cy: float, w: float, h: float) -> None:
        self.class_id = class_id
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    @classmethod
    def from_yolo_line(cls, line: str) -> Annotation:
        parts = line.strip().split()
        return cls(
            class_id=int(parts[0]),
            cx=float(parts[1]),
            cy=float(parts[2]),
            w=float(parts[3]),
            h=float(parts[4]),
        )


# ─── Synthetic frame generator ──────────────────────────────────────────────


def generate_synthetic_frame(
    width: int = 1920,
    height: int = 1080,
    num_cards: int = 7,
    num_players: int = 6,
) -> tuple[np.ndarray, list[Annotation]]:
    """
    Generate a synthetic poker frame with rendered elements and YOLO annotations.

    Args:
        width: Frame width.
        height: Frame height.
        num_cards: Number of cards to render (2 hero + up to 5 community).
        num_players: Number of player panels.

    Returns:
        (frame_bgr, annotations) tuple.
    """
    # Dark green felt table
    frame = np.full((height, width, 3), (30, 80, 40), dtype=np.uint8)

    # Draw elliptical table
    cv2.ellipse(frame, (width // 2, height // 2), (width // 3, height // 4),
                0, 0, 360, (20, 100, 50), -1)
    cv2.ellipse(frame, (width // 2, height // 2), (width // 3, height // 4),
                0, 0, 360, (60, 140, 70), 3)

    annotations: list[Annotation] = []

    # ── Cards
    ranks = "23456789TJQKA"
    suits = "♠♥♦♣"
    suit_colors = [(0, 0, 0), (0, 0, 200), (0, 0, 200), (0, 0, 0)]

    card_w, card_h = int(width * 0.04), int(height * 0.1)
    used_positions: list[tuple[int, int]] = []

    for i in range(min(num_cards, 7)):
        if i < 2:
            # Hero hole cards (bottom center)
            cx = width // 2 - card_w + i * (card_w + 10)
            cy = int(height * 0.80)
        else:
            # Community cards (center)
            cx = int(width * 0.30) + (i - 2) * (card_w + 15)
            cy = int(height * 0.42)

        # Draw card
        cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), (255, 255, 255), -1)
        cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), (100, 100, 100), 1)

        # Card text
        rank = random.choice(ranks)
        suit_idx = random.randint(0, 3)
        color = suit_colors[suit_idx]
        cv2.putText(frame, rank, (cx + 5, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, suits[suit_idx], (cx + 5, cy + card_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # YOLO annotation (normalized)
        ann = Annotation(
            class_id=YOLO_CLASSES["card"],
            cx=(cx + card_w / 2) / width,
            cy=(cy + card_h / 2) / height,
            w=card_w / width,
            h=card_h / height,
        )
        annotations.append(ann)

    # ── Pot (center)
    pot_value = random.choice([100, 250, 500, 1000, 2500])
    pot_x = int(width * 0.43)
    pot_y = int(height * 0.33)
    pot_w, pot_h = int(width * 0.14), int(height * 0.05)

    cv2.rectangle(frame, (pot_x, pot_y), (pot_x + pot_w, pot_y + pot_h), (0, 0, 0), -1)
    cv2.putText(frame, f"Pot: {pot_value}", (pot_x + 10, pot_y + pot_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    annotations.append(Annotation(
        class_id=YOLO_CLASSES["pot"],
        cx=(pot_x + pot_w / 2) / width,
        cy=(pot_y + pot_h / 2) / height,
        w=pot_w / width,
        h=pot_h / height,
    ))

    # ── Dealer button
    button_x = int(width * 0.52)
    button_y = int(height * 0.68)
    button_r = int(width * 0.012)
    cv2.circle(frame, (button_x, button_y), button_r, (255, 255, 255), -1)
    cv2.putText(frame, "D", (button_x - 7, button_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)

    annotations.append(Annotation(
        class_id=YOLO_CLASSES["dealer_button"],
        cx=button_x / width,
        cy=button_y / height,
        w=(button_r * 2) / width,
        h=(button_r * 2) / height,
    ))

    # ── Player panels
    panel_positions = [
        (0.45, 0.88), (0.72, 0.65), (0.72, 0.25),
        (0.45, 0.08), (0.18, 0.25), (0.18, 0.65),
    ]
    panel_w, panel_h = int(width * 0.12), int(height * 0.07)

    for i in range(min(num_players, 6)):
        px_frac, py_frac = panel_positions[i]
        px = int(width * px_frac)
        py = int(height * py_frac)
        stack = random.choice([500, 1000, 2000, 5000, 10000])

        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (120, 120, 120), 1)
        cv2.putText(frame, f"P{i+1}", (px + 5, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, str(stack), (px + 5, py + panel_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        annotations.append(Annotation(
            class_id=YOLO_CLASSES["player_panel"],
            cx=(px + panel_w / 2) / width,
            cy=(py + panel_h / 2) / height,
            w=panel_w / width,
            h=panel_h / height,
        ))

    return frame, annotations


# ─── PokerFrameDataset ───────────────────────────────────────────────────────


class PokerFrameDataset:
    """Manages labeled poker frame datasets in YOLO format."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def generate_synthetic(self, count: int = 100) -> int:
        """
        Generate synthetic labeled frames.

        Creates images/ and labels/ subdirectories in YOLO format.

        Returns:
            Number of frames generated.
        """
        images_dir = self.root / "images"
        labels_dir = self.root / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            frame, annotations = generate_synthetic_frame()

            # Save image
            img_path = images_dir / f"frame_{i:05d}.png"
            cv2.imwrite(str(img_path), frame)

            # Save labels
            lbl_path = labels_dir / f"frame_{i:05d}.txt"
            with open(lbl_path, "w") as f:
                for ann in annotations:
                    f.write(ann.to_yolo_line() + "\n")

        return count

    def validate(self) -> dict:
        """
        Validate dataset integrity.

        Returns:
            Dict with counts and any issues found.
        """
        images_dir = self.root / "images"
        labels_dir = self.root / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            return {"valid": False, "error": "images/ or labels/ directory missing"}

        images = set(p.stem for p in images_dir.iterdir()
                     if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        labels = set(p.stem for p in labels_dir.iterdir()
                     if p.suffix == ".txt")

        missing_labels = images - labels
        orphan_labels = labels - images

        result = {
            "valid": len(missing_labels) == 0 and len(orphan_labels) == 0,
            "total_images": len(images),
            "total_labels": len(labels),
            "missing_labels": list(missing_labels)[:10],
            "orphan_labels": list(orphan_labels)[:10],
        }
        return result

    def split(
        self,
        train: float = 0.7,
        val: float = 0.2,
        test: float = 0.1,
        seed: int = 42,
    ) -> dict[str, int]:
        """
        Split dataset into train/val/test sets.

        Creates train/, val/, test/ subdirectories each with images/ and labels/.

        Returns:
            Dict with counts per split.
        """
        images_dir = self.root / "images"
        labels_dir = self.root / "labels"

        stems = sorted(
            p.stem for p in images_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

        random.seed(seed)
        random.shuffle(stems)

        n = len(stems)
        n_train = int(n * train)
        n_val = int(n * val)

        splits = {
            "train": stems[:n_train],
            "val": stems[n_train:n_train + n_val],
            "test": stems[n_train + n_val:],
        }

        for split_name, split_stems in splits.items():
            split_img = self.root / split_name / "images"
            split_lbl = self.root / split_name / "labels"
            split_img.mkdir(parents=True, exist_ok=True)
            split_lbl.mkdir(parents=True, exist_ok=True)

            for stem in split_stems:
                # Find image file
                for ext in (".png", ".jpg", ".jpeg"):
                    src_img = images_dir / f"{stem}{ext}"
                    if src_img.exists():
                        shutil.copy2(src_img, split_img / src_img.name)
                        break

                src_lbl = labels_dir / f"{stem}.txt"
                if src_lbl.exists():
                    shutil.copy2(src_lbl, split_lbl / f"{stem}.txt")

        return {name: len(s) for name, s in splits.items()}

    def export_yolo_yaml(self, output_path: str | Path | None = None) -> Path:
        """
        Generate data.yaml for Ultralytics YOLO training.

        Args:
            output_path: Where to write the YAML (default: root/data.yaml).

        Returns:
            Path to the generated YAML file.
        """
        output_path = Path(output_path) if output_path else self.root / "data.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        root_abs = self.root.resolve()

        content = f"""# Poker Helper — YOLO Dataset Configuration
# Auto-generated by data.dataset

path: {root_abs}
train: train/images
val: val/images
test: test/images

nc: {len(YOLO_CLASS_NAMES)}
names: {YOLO_CLASS_NAMES}
"""
        output_path.write_text(content, encoding="utf-8")
        return output_path


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Poker Frame Dataset Tool")
    sub = parser.add_subparsers(dest="command")

    gen_p = sub.add_parser("generate", help="Generate synthetic frames")
    gen_p.add_argument("--count", type=int, default=100)
    gen_p.add_argument("--output", type=str, default="data/labeled_frames")

    val_p = sub.add_parser("validate", help="Validate dataset")
    val_p.add_argument("--path", type=str, required=True)

    split_p = sub.add_parser("split", help="Split dataset")
    split_p.add_argument("--path", type=str, required=True)
    split_p.add_argument("--ratio", nargs=3, type=float, default=[0.7, 0.2, 0.1])

    args = parser.parse_args()

    if args.command == "generate":
        ds = PokerFrameDataset(args.output)
        n = ds.generate_synthetic(args.count)
        print(f"Generated {n} synthetic frames in {args.output}")
        yaml_path = ds.export_yolo_yaml()
        print(f"YOLO config written to {yaml_path}")

    elif args.command == "validate":
        ds = PokerFrameDataset(args.path)
        result = ds.validate()
        print(json.dumps(result, indent=2))

    elif args.command == "split":
        ds = PokerFrameDataset(args.path)
        counts = ds.split(train=args.ratio[0], val=args.ratio[1], test=args.ratio[2])
        print(f"Split complete: {counts}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
