"""Flask demo: upload a dashcam clip, get back an annotated video.

Two models combined:
  - weights/best.pt: fine-tuned for seatbelt / no-seatbelt
  - yolo11n.pt:      pretrained, used only for class 67 ("cell phone")

Run:
    python app/server.py
Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # so `from src.infer import ...` works

from flask import Flask, render_template, request, url_for  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

from src.infer import annotate_image, annotate_video, IMAGE_EXTS, VIDEO_EXTS  # noqa: E402
from ultralytics import YOLO  # noqa: E402


SEATBELT_WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"
PHONE_WEIGHTS = PROJECT_ROOT / "yolo11n.pt"  # auto-downloaded on first YOLO() load
UPLOAD_DIR = PROJECT_ROOT / "app" / "static" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "app" / "static" / "outputs"
ALLOWED_EXTS = VIDEO_EXTS | IMAGE_EXTS
MAX_UPLOAD_MB = 200

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Load both models once at startup.
if not SEATBELT_WEIGHTS.exists():
    raise FileNotFoundError(
        f"Seatbelt weights not found at {SEATBELT_WEIGHTS}. "
        f"Pull them: gh release download --pattern 'best.pt' -D weights/"
    )
print(f"Loading seatbelt model from {SEATBELT_WEIGHTS} ...")
seatbelt_model = YOLO(str(SEATBELT_WEIGHTS))
print(f"  seatbelt classes: {seatbelt_model.names}")
print(f"Loading pretrained phone model ({PHONE_WEIGHTS.name}) ...")
phone_model = YOLO(str(PHONE_WEIGHTS))  # downloads from Ultralytics if missing


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("clip")
    if not file or file.filename == "":
        return "No file uploaded", 400

    safe = secure_filename(file.filename)
    ext = Path(safe).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTS)}", 400

    uid = uuid.uuid4().hex[:8]
    upload_path = UPLOAD_DIR / f"{uid}_{safe}"
    output_path = OUTPUT_DIR / f"{uid}_annotated{ext}"
    file.save(upload_path)

    conf = float(request.form.get("conf", "0.25"))

    if ext in VIDEO_EXTS:
        stats = annotate_video(seatbelt_model, phone_model, upload_path, output_path, conf)
    else:
        stats = annotate_image(seatbelt_model, phone_model, upload_path, output_path, conf)

    return render_template(
        "index.html",
        result_url=url_for("static", filename=f"outputs/{output_path.name}"),
        is_video=(ext in VIDEO_EXTS),
        stats=stats,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)