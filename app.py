from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import cv2
import numpy as np
import os
import math
from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
import io

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

# Only create upload folder if not running on Vercel
if not os.environ.get('VERCEL') and not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_glcm_features(gray, levels=16):
    image = np.floor(gray.astype(np.float64) / 256 * levels).astype(np.uint8)
    h, w = image.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)

    offsets = [(1, 0), (0, 1), (1, 1), (-1, 1)]

    for dy, dx in offsets:
        for y in range(h):
            for x in range(w):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    glcm[image[y, x], image[ny, nx]] += 1

    if glcm.sum() == 0:
        return 0.0, 0.0, 0.0

    glcm /= glcm.sum()

    contrast = 0.0
    homogeneity = 0.0
    energy = 0.0

    for i in range(levels):
        for j in range(levels):
            pij = glcm[i, j]
            contrast += pij * ((i - j) ** 2)
            homogeneity += pij / (1.0 + abs(i - j))
            energy += pij * pij

    return float(contrast), float(homogeneity), float(energy)


def compute_lbp(gray, P=8, R=1):
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    mean_lbp = float(lbp.mean())
    return mean_lbp


def compute_gabor(gray):
    gray = gray.astype(np.float32) / 255.0
    energies = []

    for freq in (0.1, 0.2):
        for theta in (0, np.pi/4, np.pi/2, 3*np.pi/4):
            kernel = np.real(gabor_kernel(freq, theta=theta))
            filtered = ndi.convolve(gray, kernel, mode='reflect')
            energies.append(np.mean(np.abs(filtered)))

    return float(np.mean(energies)) if energies else 0.0


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return render_template("index.html", error="File terlalu besar (maks 5MB)", result=None), 413


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Pilih gambar dulu"
        else:
            try:
                # Read image directly from memory (for Vercel compatibility)
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    error = "Gagal membaca gambar"
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    contrast, homogeneity, energy = compute_glcm_features(gray)
                    mean_lbp = compute_lbp(gray)
                    gabor_energy = compute_gabor(gray)

                    result = {
                        "contrast": round(contrast, 3),
                        "homogeneity": round(homogeneity, 3),
                        "energy": round(energy, 3),
                        "lbp": round(mean_lbp, 3),
                        "gabor": round(gabor_energy, 4),
                    }
            except Exception as e:
                error = f"Error processing image: {str(e)}"

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)