# Traffic Violation Detection (YOLO11)

Computer vision prototype that detects two common traffic violations from dashcam / CCTV footage:

1. **Seatbelt non-compliance** — driver or passenger without a fastened seatbelt
2. **Mobile phone usage** — driver holding/using a phone while driving

Built with [Ultralytics YOLO11](https://docs.ultralytics.com/), fine-tuned on public datasets from [Roboflow Universe](https://universe.roboflow.com/). A small Flask web app provides a demo UI for uploading clips and viewing annotated results.

> Status: **🚧 In active development.** See [docs/results.md](docs/results.md) for current metrics. Demo GIF will appear here once the model hits the target accuracy.

---

## Why this project

The goal is to demonstrate end-to-end ownership of a real CV pipeline — data sourcing, model fine-tuning, evaluation, and deployment — not to ship a production system.

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Model | YOLO11 (n / s) | Same easy API as YOLOv8, ~22% fewer params, better on small objects (phone in a hand). |
| Framework | PyTorch (via Ultralytics) | One-line training, well-documented, GPU-ready. |
| Datasets | Roboflow Universe | Free, pre-labeled, exportable in YOLO format. |
| Video I/O | OpenCV | Industry standard for frame-by-frame video processing. |
| Demo | Flask | Lightweight Python web server for the upload-and-detect UI. |

## Repo layout

```
data/         Raw + merged datasets (gitignored — fetch via roboflow Python pkg)
notebooks/    EDA, training, evaluation Jupyter notebooks
src/          Training script, inference script, dataset merge utility
app/          Flask demo (server + templates)
runs/         YOLO training output (gitignored — large)
weights/      Trained model weights (gitignored — fetched from GitHub Releases)
docs/         Results log, LinkedIn post drafts, screenshots
```

## Getting started (CPU dev machine)

```powershell
git clone https://github.com/<you>/traffic-violation-cv-yolo.git
cd traffic-violation-cv-yolo
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Pull the latest trained weights from GitHub Releases:
gh release download --pattern "best.pt" -D weights/
# Run inference on a sample clip:
python src/infer.py --source path/to/clip.mp4 --weights weights/best.pt
```

## Getting started (GPU training machine)

```powershell
# Same as above, then additionally install GPU PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Verify CUDA is detected:
python -c "import torch; print(torch.cuda.is_available())"
# Fetch datasets and train:
python src/merge_datasets.py
yolo train model=yolo11n.pt data=data/combined/data.yaml epochs=50 imgsz=640
```

## Documentation

- [docs/results.md](docs/results.md) — training runs, metrics, FPS, lessons learned
- [docs/linkedin_posts.md](docs/linkedin_posts.md) — public write-up drafts

## Limitations (honest list)

- Trained on public **daytime** footage only — likely degrades at night / rain / glare.
- Detects the **action** (no seatbelt, phone in hand) but does **not** track individuals across frames.
- Roboflow labels are imperfect; some class imbalance.
- Flask demo is unhardened — single user, no auth, no rate limiting.

## Credits

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) — the model framework
- [Roboflow Universe](https://universe.roboflow.com/) — open datasets (specific dataset attributions in [docs/results.md](docs/results.md))

## License

MIT
