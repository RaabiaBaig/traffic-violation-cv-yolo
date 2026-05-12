"""Stage the Roboflow seatbelt dataset into data/combined/ for YOLO training.

Why this is "seatbelt only" despite the project being about two violations:
  Phone-use detection is handled by a pretrained YOLO11 model (COCO class 67 = "cell phone")
  at inference time, not by fine-tuning. The Roboflow phone-use dataset we tried turned out
  to be classification-only (only 6 real bboxes for our target classes). Pretrained YOLO11
  already detects phones reliably in dashcam scenes, so we skip the fine-tune for that.

Usage:
    python src/merge_datasets.py --dry-run
    python src/merge_datasets.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

# Source-class names from the seatbelt Roboflow data.yaml -> unified-class names.
# None drops the source class entirely.
CLASS_MAP: dict[str, dict[str, str | None]] = {
    "seatbelt": {
        "seatbelt": "seatbelt",
        "no-seatbelt": "no_seatbelt",
    },
}

UNIFIED_CLASSES = ["seatbelt", "no_seatbelt"]
SPLITS = ["train", "valid", "test"]


def load_source_classes(source_root: Path) -> list[str]:
    yaml_path = source_root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No data.yaml found at {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text())
    names = data.get("names")
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def build_id_remap(source_classes: list[str], source_name: str) -> dict[int, int]:
    mapping = CLASS_MAP[source_name]
    remap: dict[int, int] = {}
    for src_id, src_name in enumerate(source_classes):
        if src_name not in mapping:
            raise KeyError(
                f"Class '{src_name}' from '{source_name}' is not in CLASS_MAP. "
                f"Edit src/merge_datasets.py and re-run."
            )
        target = mapping[src_name]
        if target is None:
            continue
        if target not in UNIFIED_CLASSES:
            raise ValueError(f"Target '{target}' is not in UNIFIED_CLASSES")
        remap[src_id] = UNIFIED_CLASSES.index(target)
    return remap


def merge_split(
    source_root: Path,
    source_name: str,
    remap: dict[int, int],
    split: str,
    out_root: Path,
    dry_run: bool,
) -> tuple[int, int]:
    src_images = source_root / split / "images"
    src_labels = source_root / split / "labels"
    if not src_images.exists():
        return 0, 0

    out_images = out_root / split / "images"
    out_labels = out_root / split / "labels"
    if not dry_run:
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

    n_images = n_labels = 0
    for img_path in src_images.iterdir():
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        new_name = f"{source_name}_{img_path.name}"
        if not dry_run:
            shutil.copy2(img_path, out_images / new_name)
        n_images += 1

        label_path = src_labels / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        new_lines: list[str] = []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            src_cls = int(parts[0])
            if src_cls not in remap:
                continue
            new_cls = remap[src_cls]
            new_lines.append(" ".join([str(new_cls), *parts[1:]]))
        if new_lines:
            new_label_name = f"{source_name}_{img_path.stem}.txt"
            if not dry_run:
                (out_labels / new_label_name).write_text("\n".join(new_lines) + "\n")
            n_labels += 1
    return n_images, n_labels


def write_combined_yaml(out_root: Path) -> None:
    yaml_content = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(yaml_content, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seatbelt", type=Path, default=Path("data/seatbelt"))
    parser.add_argument("--out", type=Path, default=Path("data/combined"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = {"seatbelt": args.seatbelt}

    for name, root in sources.items():
        if not root.exists():
            raise FileNotFoundError(f"{name} dataset not found at {root}")
        classes = load_source_classes(root)
        remap = build_id_remap(classes, name)
        print(f"[{name}] classes: {classes}")
        print(f"[{name}] remap (src_id -> unified_id): {remap}")

        for split in SPLITS:
            n_img, n_lbl = merge_split(root, name, remap, split, args.out, args.dry_run)
            print(f"[{name}/{split}] copied {n_img} images, {n_lbl} labels")

    if not args.dry_run:
        write_combined_yaml(args.out)
        print(f"\nWrote {args.out / 'data.yaml'} with classes: {UNIFIED_CLASSES}")
    else:
        print("\n(dry-run: no files written)")


if __name__ == "__main__":
    main()