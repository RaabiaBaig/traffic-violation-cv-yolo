"""Download seatbelt + phone-use datasets from Roboflow Universe.

One-time setup:
  1. Sign up at https://roboflow.com/ (free).
  2. Get your private API key from https://app.roboflow.com/settings/api
  3. Set it as an environment variable (PowerShell):
         $env:ROBOFLOW_API_KEY = "your_key_here"
     Or persist it for your user account:
         [System.Environment]::SetEnvironmentVariable("ROBOFLOW_API_KEY", "your_key_here", "User")
  4. Run:  python src/download_datasets.py

After the download, this script prints the class names found in each dataset's data.yaml
so you know exactly what to put in CLASS_MAP (in src/merge_datasets.py).

To swap datasets: change the entries in DATASETS below. The values come from the
Roboflow Universe URL, e.g. https://universe.roboflow.com/<workspace>/<project>/dataset/<version>.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from roboflow import Roboflow


# Dataset registry — change these to swap in different Roboflow datasets.
DATASETS = {
    "seatbelt": {
        "workspace": "seatbelttraining-7yh0f",
        "project": "seatbelt-detection-lb1ec",
        "version": 3,
    },
    "phone": {
        # Kaggle State Farm "Distracted Driver" re-published on Roboflow Universe.
        # Interior-cam shots of drivers. v7 has 9 classes including talking-on-the-phone,
        # texting, and safe-driving.
        "workspace": "oilclass",
        "project": "distracted-driver-kaggle-data",
        "version": 7,
    },
}

EXPORT_FORMAT = "yolov8"  # YOLO11 uses the same data layout as YOLOv8.
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def find_data_yaml(root: Path) -> Path | None:
    """Roboflow sometimes nests output in a versioned subfolder; find data.yaml wherever it is."""
    for candidate in root.rglob("data.yaml"):
        return candidate
    return None


def print_classes(name: str, data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text())
    names = data.get("names")
    if isinstance(names, dict):
        classes = [names[i] for i in sorted(names.keys())]
    else:
        classes = list(names or [])
    print(f"\n[{name}] data.yaml: {data_yaml}")
    print(f"[{name}] classes ({len(classes)}): {classes}")
    return classes


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print(
            "ERROR: ROBOFLOW_API_KEY is not set.\n"
            "  PowerShell (current session):  $env:ROBOFLOW_API_KEY = \"your_key_here\"\n"
            "  PowerShell (persistent):       [System.Environment]::SetEnvironmentVariable("
            "\"ROBOFLOW_API_KEY\", \"your_key_here\", \"User\")\n"
            "Get a key at https://app.roboflow.com/settings/api",
            file=sys.stderr,
        )
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    found_classes: dict[str, list[str]] = {}

    for name, info in DATASETS.items():
        out_dir = DATA_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Downloading {name}: {info['workspace']}/{info['project']} v{info['version']} ===")

        # If a data.yaml already exists, skip the download (cached).
        if find_data_yaml(out_dir) is not None:
            print(f"  cached — data.yaml already present, skipping download")
        else:
            try:
                workspace = rf.workspace(info["workspace"])
                project = workspace.project(info["project"])
                try:
                    version = project.version(info["version"])
                except RuntimeError as ver_err:
                    # Help the user find a working version number.
                    print(f"  version {info['version']} not found: {ver_err}")
                    available = project.versions()
                    print(f"  available versions on this project: "
                          f"{[v.version for v in available] if available else 'NONE'}")
                    if not available:
                        continue
                    version = available[0]  # newest is typically first
                    print(f"  falling back to latest: {version.version}")
                print(f"  downloading to {out_dir} (format={EXPORT_FORMAT}) ...")
                # overwrite=True is critical: the SDK silently no-ops if the dir exists.
                version.download(EXPORT_FORMAT, location=str(out_dir), overwrite=True)
            except Exception as exc:  # noqa: BLE001 — we want to see anything Roboflow throws
                import traceback
                print(f"  ERROR: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                continue

        files_after = list(out_dir.rglob("*"))
        print(f"  files in {out_dir}: {len(files_after)}")
        if files_after:
            print(f"  examples: {[str(p.relative_to(out_dir)) for p in files_after[:5]]}")

        yaml_path = find_data_yaml(out_dir)
        if yaml_path is None:
            print(f"  WARNING: no data.yaml found under {out_dir}")
            continue
        found_classes[name] = print_classes(name, yaml_path)

    print("\n" + "=" * 60)
    print("NEXT STEP: open src/merge_datasets.py and update CLASS_MAP")
    print("so every class name above is mapped to a unified class.")
    print("Then run:  python src/merge_datasets.py --dry-run")
    print("=" * 60)


if __name__ == "__main__":
    main()