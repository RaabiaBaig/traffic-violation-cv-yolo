"""Flask demo: upload a dashcam clip, get back an annotated video.

Run:
    python app/server.py
Then open http://localhost:5000 in your browser.

Place a trained model at weights/best.pt before starting the server.
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
WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"
UPLOAD_DIR = PROJECT_ROOT / "app" / "static" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "app" / "static" / "outputs"
ALLOWED_EXTS = VIDEO_EXTS | IMAGE_EXTS
MAX_UPLOAD_MB = 200

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Load the model once at startup so each request is fast.
if not WEIGHTS.exists():
    raise FileNotFoundError(
        f"Trained weights not found at {WEIGHTS}. "
        f"Pull them with: gh release download --pattern 'best.pt' -D weights/"
    )
print(f"Loading model from {WEIGHTS} ...")
model = YOLO(str(WEIGHTS))
print(f"Model classes: {model.names}")


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
        stats = annotate_video(model, upload_path, output_path, conf)
    else:
        stats = annotate_image(model, upload_path, output_path, conf)

    return render_template(
        "index.html",
        result_url=url_for("static", filename=f"outputs/{output_path.name}"),
        is_video=(ext in VIDEO_EXTS),
        stats=stats,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)