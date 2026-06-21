from flask import Flask, render_template, request
import cv2
import numpy as np

app = Flask(__name__)


# =========================
# ANALISIS WARNA & KONDISI
# =========================
def analyze_meat(img):
    # RGB
    avg_color = img.mean(axis=(0, 1))
    b, g, r = avg_color

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv.mean(axis=(0, 1))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    # =========================
    # SIMPLE TEXTURE (EDGE DENSITY)
    # =========================
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges)

    # =========================
    # BLOBS / BERCak
    # =========================
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(clean)
    blob_count = max(0, num_labels - 1)

    # =========================
    # SCORING SYSTEM
    # =========================
    score = 0
    explanation = []

    # WARNA (daging segar cenderung merah)
    if r > g and r > b:
        score += 1
        explanation.append("Warna merah dominan (indikasi segar)")

    # HUE
    if 0 <= h <= 50:
        score += 1
        explanation.append("Hue masih dalam range daging segar")

    # SATURASI
    if s > 40:
        score += 1
        explanation.append("Saturasi cukup baik")

    # BRIGHTNESS
    if 80 <= brightness <= 160:
        score += 1
        explanation.append("Kecerahan normal")

    # TEXTURE
    if edge_density < 50:
        score += 1
        explanation.append("Tekstur tidak terlalu kasar")

    # BERCak
    if blob_count < 5:
        score += 1
        explanation.append("Tidak banyak bercak")

    # =========================
    # FINAL STATUS
    # =========================
    if score >= 5:
        status = "Segar"
    elif score >= 3:
        status = "Kurang Segar"
    else:
        status = "Tidak Segar"

    return {
        "status": status,
        "score": score,
        "explanation": explanation,

        "rgb": {"r": float(r), "g": float(g), "b": float(b)},
        "hsv": {"h": float(h), "s": float(s), "v": float(v)},
        "brightness": float(brightness),
        "edge_density": float(edge_density),
        "blob_count": int(blob_count)
    }


# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file:
            error = "Silakan upload gambar."
        else:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                error = "Gambar tidak valid."
            else:
                result = analyze_meat(img)

    return render_template("index.html", result=result, error=error)


# =========================
# VERCEL ENTRYPOINT
# =========================
app = app
