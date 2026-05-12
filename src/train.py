"""Fine-tune a YOLO11 model on the combined traffic-violation dataset.

Run on the GPU laptop:
    python src/train.py
    python src/train.py --model yolo11s.pt --epochs 100 --imgsz 640
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Base weights. Try yolo11n.pt (fast) or yolo11s.pt (more accurate).")
    parser.add_argument("--data", type=Path, default=Path("data/combined/data.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16,
                        help="Lower this if you hit CUDA out-of-memory errors.")
    parser.add_argument("--name", default="baseline",
                        help="Run name. Outputs land in runs/detect/<name>/.")
    parser.add_argument("--device", default="0",
                        help="GPU id, or 'cpu'. Use '0' on a single-GPU machine.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping: stop if val mAP hasn't improved in N epochs.")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} not found. Did you run src/merge_datasets.py first?"
        )

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        plots=True,
    )


if __name__ == "__main__":
    main()