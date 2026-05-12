"""Run a trained YOLO model on a video or image and save an annotated output.

Examples:
    python src/infer.py --source clip.mp4 --weights weights/best.pt
    python src/infer.py --source photo.jpg --weights weights/best.pt --conf 0.4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def annotate_video(model: YOLO, src: Path, dst: Path, conf: float) -> dict:
    """Run YOLO on every frame, write an annotated MP4. Returns simple stats."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))

    detections_per_class: dict[str, int] = {}
    frame_idx = 0
    t0 = time.time()

    # stream=True yields one Results object per frame without buffering the whole video.
    try:
        for result in model.predict(source=str(src), stream=True, conf=conf, verbose=False):
            annotated = result.plot()  # numpy array with boxes already drawn
            writer.write(annotated)
            frame_idx += 1
            if result.boxes is not None:
                for cls_id in result.boxes.cls.tolist():
                    name = model.names[int(cls_id)]
                    detections_per_class[name] = detections_per_class.get(name, 0) + 1
            if frame_idx % 50 == 0:
                pct = (frame_idx / total_frames * 100) if total_frames else 0
                print(f"  frame {frame_idx}/{total_frames} ({pct:.0f}%)")
    finally:
        writer.release()
        cap.release()

    elapsed = time.time() - t0
    return {
        "frames": frame_idx,
        "elapsed_sec": round(elapsed, 2),
        "fps": round(frame_idx / elapsed, 1) if elapsed > 0 else 0,
        "detections": detections_per_class,
    }


def annotate_image(model: YOLO, src: Path, dst: Path, conf: float) -> dict:
    results = model.predict(source=str(src), conf=conf, verbose=False)
    annotated = results[0].plot()
    cv2.imwrite(str(dst), annotated)
    detections: dict[str, int] = {}
    if results[0].boxes is not None:
        for cls_id in results[0].boxes.cls.tolist():
            name = model.names[int(cls_id)]
            detections[name] = detections.get(name, 0) + 1
    return {"detections": detections}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Image or video file")
    parser.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path. Defaults to <source>_annotated.<ext> next to source.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (0-1). Lower = more boxes, more false positives.")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(args.source)
    if not args.weights.exists():
        raise FileNotFoundError(
            f"{args.weights} not found. Pull from GitHub Releases: "
            f"gh release download --pattern 'best.pt' -D weights/"
        )

    ext = args.source.suffix.lower()
    if args.out is None:
        args.out = args.source.with_name(f"{args.source.stem}_annotated{ext}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.weights}")
    model = YOLO(str(args.weights))

    print(f"Running inference on {args.source} -> {args.out}")
    if ext in VIDEO_EXTS:
        stats = annotate_video(model, args.source, args.out, args.conf)
    elif ext in IMAGE_EXTS:
        stats = annotate_image(model, args.source, args.out, args.conf)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    print("\nDone.")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()