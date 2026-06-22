from flask import Flask, render_template, request
import base64
import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# =====================
# ANALISIS WARNA
# =====================
def color_analysis(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    r, g, b = cv2.split(rgb)
    h, s, v = cv2.split(hsv)
    l_lab, a_lab, b_lab = cv2.split(lab)

    hue = float(np.mean(h) * 2)
    saturation = float((np.mean(s) / 255) * 100)
    value = float((np.mean(v) / 255) * 100)
    lightness = float((np.mean(l_lab) / 255) * 100)
    a_value = float(np.mean(a_lab) - 128)
    b_value = float(np.mean(b_lab) - 128)

    if 85 <= hue <= 155:
        hue_status = "Mentah"
        status = "Mentah"
    elif 15 <= hue <= 55:
        hue_status = "Matang"
        status = "Matang"
    else:
        hue_status = "Setengah Matang"
        status = "Setengah Matang"

    if saturation < 25:
        saturation_status = "Kesepakatan warna rendah / pucat"
    elif saturation < 55:
        saturation_status = "Kesepakatan warna sedang"
    else:
        saturation_status = "Kesepakatan warna tinggi"

    if value < 35:
        brightness_status = "Gelap"
    elif value < 70:
        brightness_status = "Kecerahan normal"
    else:
        brightness_status = "Terlalu cerah"

    return {
        "rgb": {
            "r": round(float(np.mean(r)), 2),
            "g": round(float(np.mean(g)), 2),
            "b": round(float(np.mean(b)), 2)
        },
        "hsv": {
            "h": round(hue, 2),
            "s": round(saturation, 2),
            "v": round(value, 2)
        },
        "lab": {
            "l": round(lightness, 2),
            "a": round(a_value, 2),
            "b": round(b_value, 2)
        },
        "h": round(hue, 2),
        "s": round(saturation, 2),
        "v": round(value, 2),
        "hue_status": hue_status,
        "saturation_status": saturation_status,
        "brightness_status": brightness_status,
        "state": status
    }


# =====================
# ANALISIS TEKSTUR
# =====================
def texture_analysis(gray, meat_mask=None):
    small_gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    quantized = (small_gray / 32).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True
    )

    contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    energy = float(np.mean(graycoprops(glcm, 'energy')))

    lbp = local_binary_pattern(small_gray, 8, 1, method='uniform')
    lbp_variance = float(np.var(lbp))

    gabor_values = []
    for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(small_gray, cv2.CV_32F, kernel)
        gabor_values.append(float(np.mean(np.abs(filtered))))
    gabor_pattern = float(np.mean(gabor_values))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dark_spots = cv2.inRange(blur, 0, 75)
    edges = cv2.Canny(gray, 70, 160)

    if meat_mask is not None:
        dark_spots = cv2.bitwise_and(dark_spots, dark_spots, mask=meat_mask)
        edges = cv2.bitwise_and(edges, edges, mask=meat_mask)
        mask_area = max(int(cv2.countNonZero(meat_mask)), 1)
    else:
        mask_area = max(gray.shape[0] * gray.shape[1], 1)

    spot_ratio = float((cv2.countNonZero(dark_spots) / mask_area) * 100)
    wrinkle_ratio = float((cv2.countNonZero(edges) / mask_area) * 100)

    if spot_ratio > 8 or wrinkle_ratio > 14:
        surface_status = "Bintik/kerutan tinggi, indikasi terlalu matang"
        state = "Terlalu Matang"
    elif contrast < 2.5 and homogeneity > 0.65:
        surface_status = "Permukaan halus, indikasi mentah/matang normal"
        state = "Mentah / Matang"
    else:
        surface_status = "Tekstur sedang, indikasi setengah matang"
        state = "Setengah Matang"

    return {
        "contrast": round(contrast, 2),
        "homogeneity": round(homogeneity, 2),
        "energy": round(energy, 2),
        "lbp": round(lbp_variance, 2),
        "gabor": round(gabor_pattern, 2),
        "spot_ratio": round(spot_ratio, 2),
        "wrinkle_ratio": round(wrinkle_ratio, 2),
        "surface_status": surface_status,
        "state": state
    }


# =====================
# ANALISIS BENTUK
# =====================
def get_meat_mask(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if cv2.countNonZero(mask) > (mask.size * 0.75):
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def detect_blobs(gray, mask):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, dark = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    dark = cv2.bitwise_and(dark, dark, mask=mask)

    contours, _ = cv2.findContours(
        dark,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    blobs = [c for c in contours if 12 <= cv2.contourArea(c) <= 2500]
    blob_area = sum(cv2.contourArea(c) for c in blobs)
    meat_area = max(cv2.countNonZero(mask), 1)

    return len(blobs), float((blob_area / meat_area) * 100)


def shape_analysis(gray, mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return {
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "area": 0,
            "perimeter": 0,
            "aspect_ratio": 0,
            "convexity": 0,
            "roundness": 0,
            "blob_count": 0,
            "blob_area_ratio": 0,
            "state": "Tidak Terdeteksi"
        }

    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    x, y, w, h = cv2.boundingRect(c)

    aspect_ratio = w / h if h != 0 else 0

    hull = cv2.convexHull(c)

    hull_area = cv2.contourArea(hull)

    convexity = area / hull_area if hull_area != 0 else 0

    roundness = (
        (4 * np.pi * area) /
        (perimeter * perimeter)
    ) if perimeter != 0 else 0

    blob_count, blob_area_ratio = detect_blobs(gray, mask)

    if convexity < 0.82:
        shape_status = "Ada indikasi penyok / perubahan bentuk"
    elif roundness < 0.45:
        shape_status = "Bentuk memanjang / tidak beraturan"
    else:
        shape_status = "Bentuk stabil"

    if blob_count >= 8 or blob_area_ratio > 6:
        stain_status = "Bercak terdeteksi cukup banyak"
    elif blob_count > 0:
        stain_status = "Bercak kecil terdeteksi"
    else:
        stain_status = "Bercak tidak dominan"

    if convexity < 0.82 or blob_area_ratio > 6:
        state = "Terlalu Matang / Perubahan Bentuk"
    elif 0.45 <= roundness <= 1.05:
        state = "Mentah / Matang Normal"
    else:
        state = "Perlu Pemeriksaan"

    return {
        "bounding_box": {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        },
        "area": round(float(area), 2),
        "perimeter": round(float(perimeter), 2),
        "aspect_ratio": round(float(aspect_ratio), 2),
        "convexity": round(float(convexity), 2),
        "roundness": round(float(roundness), 2),
        "blob_count": int(blob_count),
        "blob_area_ratio": round(float(blob_area_ratio), 2),
        "shape_status": shape_status,
        "stain_status": stain_status,
        "state": state
    }


def final_decision(color, texture, shape):
    scores = {
        "Mentah": 0,
        "Setengah Matang": 0,
        "Matang": 0,
        "Terlalu Matang": 0
    }

    if color["state"] == "Mentah":
        scores["Mentah"] += 2
    elif color["state"] == "Matang":
        scores["Matang"] += 2
    else:
        scores["Setengah Matang"] += 2

    if texture["state"] == "Terlalu Matang":
        scores["Terlalu Matang"] += 2
    elif texture["state"] == "Setengah Matang":
        scores["Setengah Matang"] += 1
    else:
        scores[color["state"]] += 1

    if "Terlalu Matang" in shape["state"]:
        scores["Terlalu Matang"] += 2
    elif "Normal" in shape["state"]:
        scores[color["state"]] += 1

    return max(scores, key=scores.get)


# =====================
# HOME
# =====================
@app.route("/")
def home():
    return render_template("index.html")


# =====================
# UPLOAD
# =====================
@app.route("/upload", methods=["POST"])
def upload():

    if "image" not in request.files:
        return "File tidak ditemukan"

    file = request.files["image"]

    data = np.frombuffer(file.read(), np.uint8)
    mime_type = file.mimetype or "image/jpeg"
    uploaded_image = f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        return "Gagal membaca gambar"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    meat_mask = get_meat_mask(gray)

    color = color_analysis(img)

    texture = texture_analysis(gray, meat_mask)

    shape = shape_analysis(gray, meat_mask)

    final = final_decision(color, texture, shape)

    return render_template(
        "index.html",
        color=color,
        texture=texture,
        shape=shape,
        final=final,
        uploaded_image=uploaded_image
    )


application = app

if __name__ == "__main__":
    app.run(debug=True)
