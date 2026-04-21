"""
YOLO11 Training — CLI utility for training poker object detection model.

Usage:
    python -m services.vision_core.train --data data/labeled_frames/data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def train(
    data_yaml: str,
    model: str = "yolo11n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "cpu",
    project: str = "runs/train",
    name: str = "poker",
) -> Path:
    """
    Train YOLO11 model on poker dataset.

    Args:
        data_yaml: Path to data.yaml configuration.
        model: Base model to fine-tune.
        epochs: Number of training epochs.
        batch_size: Batch size.
        img_size: Input image size.
        device: Training device ('cpu', '0', '0,1', etc.).
        project: Project directory for training results.
        name: Experiment name.

    Returns:
        Path to best trained weights.
    """
    from ultralytics import YOLO

    yolo = YOLO(model)

    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
    )

    best_path = Path(project) / name / "weights" / "best.pt"
    logger.info("Training complete. Best weights: %s", best_path)
    return best_path


def export_model(
    weights_path: str,
    format: str = "onnx",
    img_size: int = 640,
) -> Path:
    """
    Export trained model to deployment format.

    Args:
        weights_path: Path to .pt weights.
        format: Export format ('onnx', 'torchscript', 'openvino', etc.).
        img_size: Image size for export.

    Returns:
        Path to exported model.
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    path = model.export(format=format, imgsz=img_size)
    logger.info("Model exported to %s: %s", format, path)
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11 Poker Model Training")

    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Train model")
    train_p.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    train_p.add_argument("--model", type=str, default="yolo11n.pt", help="Base model")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--batch", type=int, default=16)
    train_p.add_argument("--imgsz", type=int, default=640)
    train_p.add_argument("--device", type=str, default="cpu")

    export_p = sub.add_parser("export", help="Export model")
    export_p.add_argument("--weights", type=str, required=True)
    export_p.add_argument("--format", type=str, default="onnx")
    export_p.add_argument("--imgsz", type=int, default=640)

    args = parser.parse_args()

    if args.command == "train":
        best = train(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=args.device,
        )
        print(f"✅ Training complete. Best weights: {best}")

    elif args.command == "export":
        path = export_model(args.weights, format=args.format, img_size=args.imgsz)
        print(f"✅ Exported to: {path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
