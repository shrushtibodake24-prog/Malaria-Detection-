from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from app.utils import predict_image


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def _recommendation_message(label: str) -> str:
    """
    Non-clinical, educational-only recommendations and disclaimers.
    Must update dynamically based on prediction.
    """
    general = (
        "Educational use only — not a medical device. Do not use this result for diagnosis. "
        "Confirm with laboratory testing and consult a qualified medical professional."
    )
    if label == "Parasitized":
        return (
            "This image was classified as Parasitized. "
            "Please seek guidance from a healthcare professional and ensure proper diagnosis using laboratory confirmation. "
            + general
        )
    # Uninfected
    return (
        "This image was classified as Uninfected. "
        "If symptoms persist or you have concerns, consult a healthcare professional and consider laboratory confirmation. "
        + general
    )


def create_app() -> Flask:
    base_dir = Path(__file__).resolve().parent
    template_dir = base_dir / "app" / "templates"
    static_dir = base_dir / "app" / "static"
    upload_dir = static_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    flask_app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
    flask_app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB
    flask_app.config["UPLOAD_FOLDER"] = str(upload_dir)
    flask_app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

    @flask_app.get("/")
    def index():
        return render_template(
            "index.html",
            label=None,
            confidence=None,
            image_url=None,
            recommendation=None,
            error=None,
        )

    @flask_app.post("/predict")
    def predict():
        try:
            if "image" not in request.files:
                return render_template("index.html", error="No file part in the request.")

            file = request.files["image"]
            if not file or file.filename is None or file.filename.strip() == "":
                return render_template("index.html", error="No file selected.")

            if not _allowed_file(file.filename):
                return render_template(
                    "index.html",
                    error="Invalid file type. Please upload an image (.png, .jpg, .jpeg, .bmp, .webp).",
                )

            filename = secure_filename(file.filename)
            # Avoid collisions by prefixing with a simple counter/unique name
            save_path = Path(flask_app.config["UPLOAD_FOLDER"]) / filename
            if save_path.exists():
                stem, suffix = save_path.stem, save_path.suffix
                i = 1
                while True:
                    candidate = save_path.with_name(f"{stem}_{i}{suffix}")
                    if not candidate.exists():
                        save_path = candidate
                        break
                    i += 1

            file.save(str(save_path))

            # Run prediction
            label, confidence = predict_image(str(save_path))
            recommendation = _recommendation_message(label)

            image_url = f"{flask_app.static_url_path}/uploads/{save_path.name}"
            return render_template(
                "index.html",
                label=label,
                confidence=confidence,
                image_url=image_url,
                recommendation=recommendation,
                error=None,
            )
        except Exception:
            # Avoid leaking internal errors to the user; keep it safe for production.
            return render_template(
                "index.html",
                label=None,
                confidence=None,
                image_url=None,
                recommendation=None,
                error="Prediction failed. Please upload a valid image file and try again.",
            )

    return flask_app


if __name__ == "__main__":
    app = create_app()
    # Accessible on localhost; change host to 0.0.0.0 if needed.
    app.run(host="127.0.0.1", port=5000, debug=True)

