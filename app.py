from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import cv2
import numpy as np
import os
import math
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB max upload

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_glcm_features(gray, levels=16):
    # reduce levels to speed up GLCM
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


def compute_lbp_feature(gray, P=8, R=1):
    # compute LBP and return proportion of uniform patterns and mean LBP
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    lbp = lbp.astype(np.float32)
    mean_lbp = float(lbp.mean())
    # proportion of zero (smooth) patterns
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
    uniform_prop = float(hist[0]) if hist.size > 0 else 0.0
    return mean_lbp, uniform_prop


def compute_gabor_features(gray, frequencies=(0.1, 0.2), thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    # apply several Gabor kernels and compute mean energy
    gray_f = gray.astype(np.float32) / 255.0
    energies = []
    for freq in frequencies:
        for theta in thetas:
            kernel = np.real(gabor_kernel(freq, theta=theta))
            filtered = ndi.convolve(gray_f, kernel, mode='reflect')
            energy = np.mean(np.abs(filtered))
            energies.append(float(energy))
    if not energies:
        return 0.0
    return float(np.mean(energies))


def compute_color_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_lab = lab.mean(axis=(0, 1))
    l = float(avg_lab[0])
    a = float(avg_lab[1])
    b = float(avg_lab[2])
    return l, a, b


def compute_shape_features(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            'area': 0,
            'perimeter': 0,
            'compactness': 0.0,
            'bbox': (0, 0, 0, 0),
            'aspect_ratio': 0.0,
            'convexity': 0.0,
            'roundness': 0.0,
        }

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    compactness = 0.0
    if perimeter > 0:
        compactness = 4 * math.pi * area / (perimeter * perimeter)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = float(area) / float(hull_area) if hull_area > 0 else 0.0
    roundness = float(4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0

    return {
        'area': int(area),
        'perimeter': int(perimeter),
        'compactness': float(compactness),
        'bbox': (int(x), int(y), int(w), int(h)),
        'aspect_ratio': float(aspect_ratio),
        'convexity': float(convexity),
        'roundness': float(roundness),
    }


def detect_blobs(gray):
    # simple blob detection using connected components after morphological opening
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened)
    # exclude background
    blob_count = int(num_labels - 1) if num_labels > 0 else 0
    areas = [int(s[cv2.CC_STAT_AREA]) for s in stats[1:]] if num_labels > 1 else []
    mean_blob_area = float(np.mean(areas)) if areas else 0.0
    return blob_count, mean_blob_area


def classify_freshness(color, texture, shape, lbp_info, gabor_energy, blob_info):
    r, g, b, h, s, v, l_lab, a_lab, b_lab = color
    contrast, homogeneity, energy = texture
    area = shape.get('area', 0)
    compactness = shape.get('compactness', 0.0)
    aspect_ratio = shape.get('aspect_ratio', 0.0)
    convexity = shape.get('convexity', 0.0)
    roundness = shape.get('roundness', 0.0)

    mean_lbp, uniform_prop = lbp_info
    blob_count, mean_blob_area = blob_info

    score = 0
    explanation = []

    # Color heuristics: raw meat tends to have higher hue around red-pink (H approx 0-10 or high around 160-180)
    # cooked/browned meat shifts hue lower (towards orange/brown). These are rough heuristics.
    if (0 <= h <= 20 or h >= 160) and s > 40 and v > 50:
        score += 1
        explanation.append("Warna (HSV) mendukung kondisi segar/mentah (nilai Hue dan Saturation cocok).")
    else:
        explanation.append("Warna menunjukkan indikasi perubahan (coklat/gelap) yang mungkin menandakan kematangan atau degradasi.")

    # Lab can detect browning: larger 'a' and 'b' shifts
    if l_lab > 50 and abs(a_lab - 128) < 15:
        explanation.append("Kecerahan Lab menunjukkan daging tidak terlalu gelap.")

    # Texture: low contrast + high homogeneity and energy suggests uniform surface (fresh)
    if contrast < 60 and homogeneity > 0.5 and energy > 0.12:
        score += 1
        explanation.append("Tekstur (GLCM) relatif homogen, mendukung kesegaran.")
    else:
        explanation.append("Tekstur menunjukkan variasi/kerutan yang mungkin terkait kematangan atau bercak.")

    # LBP/Gabor: smooth surfaces (low mean LBP) indicate fresh meat; high gabor energy may indicate fine patterns
    if mean_lbp < 2.5:
        score += 1
        explanation.append("LBP menunjukkan pola halus (permukaan relatif halus).")
    else:
        explanation.append("LBP menunjukkan banyak tekstur/bintik, bisa menandakan bercak atau kematangan.")

    if gabor_energy < 0.03:
        explanation.append("Gabor filter tidak mendeteksi pola berlebih pada permukaan.")
    else:
        explanation.append("Gabor menunjukkan pola tekstur halus yang terdeteksi.")

    # Shape and blobs
    if area >= 500 and compactness >= 0.02 and convexity > 0.8:
        score += 1
        explanation.append("Bentuk objek cukup besar, kompak, dan konveksitas baik.")
    else:
        explanation.append("Bentuk/pemotongan tidak ideal atau perubahan bentuk terdeteksi.")

    if blob_count > 3:
        explanation.append(f"Terdeteksi {blob_count} bercak/area; ini perlu pemeriksaan visual lebih lanjut.")

    if score >= 3:
        status = "Segar"
    elif score == 2:
        status = "Kurang Segar"
    else:
        status = "Tidak Segar"

    return status, explanation


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    error = "File terlalu besar (maks 5 MB). Silakan gunakan file yang lebih kecil."
    return render_template("index.html", result=None, error=error), 413


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Silakan pilih file gambar terlebih dahulu."
        else:
            filename = secure_filename(file.filename)
            if not ("." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS):
                error = "Format file tidak didukung. Gunakan PNG, JPG, JPEG, BMP, atau GIF."
            else:
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                img = cv2.imread(filepath)
                if img is None:
                    error = "Gagal memuat gambar. Pastikan file gambar valid."
                else:
                    # Color: RGB, HSV, Lab
                    avg_color = img.mean(axis=(0, 1))
                    b = int(avg_color[0])
                    g = int(avg_color[1])
                    r = int(avg_color[2])

                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    avg_hsv = hsv.mean(axis=(0, 1))
                    h = float(avg_hsv[0])
                    s = float(avg_hsv[1])
                    v = float(avg_hsv[2])

                    l_lab, a_lab, b_lab = compute_color_lab(img)

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    contrast, homogeneity, energy = compute_glcm_features(gray)

                    # texture features: LBP and Gabor
                    mean_lbp, uniform_prop = compute_lbp_feature(gray)
                    gabor_energy = compute_gabor_features(gray)

                    # shape features and blobs
                    shape_feats = compute_shape_features(gray)
                    blob_count, mean_blob_area = detect_blobs(gray)

                    status, explanation = classify_freshness(
                        (r, g, b, h, s, v, l_lab, a_lab, b_lab),
                        (contrast, homogeneity, energy),
                        shape_feats,
                        (mean_lbp, uniform_prop),
                        gabor_energy,
                        (blob_count, mean_blob_area),
                    )

                    result = {
                        "filename": filename,
                        "r": r,
                        "g": g,
                        "b": b,
                        "h": round(h, 1),
                        "s": round(s, 1),
                        "v": round(v, 1),
                        "l_lab": round(l_lab, 1),
                        "a_lab": round(a_lab, 1),
                        "b_lab": round(b_lab, 1),
                        "contrast": round(contrast, 2),
                        "homogeneity": round(homogeneity, 3),
                        "energy": round(energy, 3),
                        "mean_lbp": round(mean_lbp, 3),
                        "uniform_lbp": round(uniform_prop, 3),
                        "gabor_energy": round(gabor_energy, 4),
                        "area": shape_feats.get('area', 0),
                        "perimeter": shape_feats.get('perimeter', 0),
                        "compactness": round(shape_feats.get('compactness', 0.0), 3),
                        "bbox": shape_feats.get('bbox', (0, 0, 0, 0)),
                        "aspect_ratio": round(shape_feats.get('aspect_ratio', 0.0), 3),
                        "convexity": round(shape_feats.get('convexity', 0.0), 3),
                        "roundness": round(shape_feats.get('roundness', 0.0), 3),
                        "blob_count": blob_count,
                        "mean_blob_area": round(mean_blob_area, 1),
                        "status": status,
                        "explanation": explanation,
                    }

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
