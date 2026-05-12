"""Run two-model inference on a video or image and save an annotated output.

Two models combined:
  - weights/best.pt: fine-tuned for seatbelt vs no-seatbelt (custom training)
  - yolo11n.pt:      pretrained on COCO; we use only class 67 ("cell phone")

Examples:
    python src/infer.py --source clip.mp4
    python src/infer.py --source photo.jpg --conf 0.4
    python src/infer.py --source clip.mp4 --seatbelt-weights weights/best.pt --phone-weights yolo11n.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# COCO class id for "cell phone" in pretrained YOLO11.
PHONE_CLASS_ID = 67

# BGR colors (OpenCV convention).
COLOR_SEATBELT = (0, 200, 0)       # green   — compliant
COLOR_NO_SEATBELT = (0, 0, 255)    # red     — seatbelt violation
COLOR_PHONE = (0, 140, 255)        # orange  — phone violation


def _draw_label(frame, x1: int, y1: int, x2: int, y2: int, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, max(th, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_frame(frame, seatbelt_model: YOLO, phone_model: YOLO, conf: float) -> tuple:
    """Run both models on a single BGR frame. Returns (annotated_frame, counts)."""
    counts: dict[str, int] = {}
    out = frame.copy()

    sb_res = seatbelt_model.predict(frame, conf=conf, verbose=False)[0]
    if sb_res.boxes is not None:
        for box in sb_res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_name = sb_res.names[int(box.cls[0])]
            confv = float(box.conf[0])
            color = COLOR_SEATBELT if cls_name == "seatbelt" else COLOR_NO_SEATBELT
            _draw_label(out, x1, y1, x2, y2, f"{cls_name} {confv:.2f}", color)
            counts[cls_name] = counts.get(cls_name, 0) + 1

    ph_res = phone_model.predict(frame, conf=conf, classes=[PHONE_CLASS_ID], verbose=False)[0]
    if ph_res.boxes is not None:
        for box in ph_res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confv = float(box.conf[0])
            _draw_label(out, x1, y1, x2, y2, f"phone_use {confv:.2f}", COLOR_PHONE)
            counts["phone_use"] = counts.get("phone_use", 0) + 1

    return out, counts


def annotate_video(seatbelt_model: YOLO, phone_model: YOLO, src: Path, dst: Path, conf: float) -> dict:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))

    detections: dict[str, int] = {}
    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, counts = annotate_frame(frame, seatbelt_model, phone_model, conf)
            writer.write(annotated)
            frame_idx += 1
            for k, v in counts.items():
                detections[k] = detections.get(k, 0) + v
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
        "detections": detections,
    }


def annotate_image(seatbelt_model: YOLO, phone_model: YOLO, src: Path, dst: Path, conf: float) -> dict:
    frame = cv2.imread(str(src))
    if frame is None:
        raise RuntimeError(f"Could not read image: {src}")
    annotated, counts = annotate_frame(frame, seatbelt_model, phone_model, conf)
    cv2.imwrite(str(dst), annotated)
    return {"detections": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Image or video file")
    parser.add_argument("--seatbelt-weights", type=Path, default=Path("weights/best.pt"),
                        help="Fine-tuned seatbelt model.")
    parser.add_argument("--phone-weights", type=Path, default=Path("yolo11n.pt"),
                        help="Pretrained model used for cell-phone detection. "
                             "Auto-downloads from Ultralytics on first use.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(args.source)
    if not args.seatbelt_weights.exists():
        raise FileNotFoundError(
            f"{args.seatbelt_weights} not found. Pull from GitHub Releases: "
            f"gh release download --pattern 'best.pt' -D weights/"
        )

    ext = args.source.suffix.lower()
    if args.out is None:
        args.out = args.source.with_name(f"{args.source.stem}_annotated{ext}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading seatbelt model: {args.seatbelt_weights}")
    seatbelt_model = YOLO(str(args.seatbelt_weights))
    print(f"Loading phone model:    {args.phone_weights} (filtered to class {PHONE_CLASS_ID} = cell phone)")
    phone_model = YOLO(str(args.phone_weights))

    print(f"Inference: {args.source} -> {args.out}")
    if ext in VIDEO_EXTS:
        stats = annotate_video(seatbelt_model, phone_model, args.source, args.out, args.conf)
    elif ext in IMAGE_EXTS:
        stats = annotate_image(seatbelt_model, phone_model, args.source, args.out, args.conf)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    print("\nDone.")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()