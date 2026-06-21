from flask import Flask, render_template, request
import cv2
import numpy as np
import math
from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

app = Flask(__name__)

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

    contrast = 0
    homogeneity = 0
    energy = 0

    for i in range(levels):
        for j in range(levels):
            pij = glcm[i, j]

            contrast += pij * ((i - j) ** 2)
            homogeneity += pij / (1 + abs(i - j))
            energy += pij * pij

    return float(contrast), float(homogeneity), float(energy)


def compute_lbp_feature(gray):
    lbp = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method="uniform"
    )

    mean_lbp = float(lbp.mean())

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        density=True
    )

    uniform_prop = float(hist[0]) if len(hist) > 0 else 0

    return mean_lbp, uniform_prop


def compute_gabor_features(gray):

    gray_f = gray.astype(np.float32) / 255.0

    energies = []

    frequencies = [0.1, 0.2]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    for freq in frequencies:
        for theta in thetas:

            kernel = np.real(
                gabor_kernel(freq, theta=theta)
            )

            filtered = ndi.convolve(
                gray_f,
                kernel,
                mode="reflect"
            )

            energies.append(
                float(np.mean(np.abs(filtered)))
            )

    return float(np.mean(energies))


def compute_color_lab(img):

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    avg = lab.mean(axis=(0, 1))

    return (
        float(avg[0]),
        float(avg[1]),
        float(avg[2])
    )
    def compute_shape_features(gray):

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {
            "area": 0,
            "perimeter": 0,
            "aspect_ratio": 0,
            "convexity": 0,
            "roundness": 0
        }

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = (
        float(w) / float(h)
        if h > 0 else 0
    )

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    convexity = (
        float(area) / float(hull_area)
        if hull_area > 0 else 0
    )

    roundness = (
        (4 * math.pi * area)
        / (perimeter * perimeter)
        if perimeter > 0 else 0
    )

    return {
        "area": int(area),
        "perimeter": int(perimeter),
        "aspect_ratio": round(aspect_ratio, 3),
        "convexity": round(convexity, 3),
        "roundness": round(roundness, 3)
    }


def detect_blobs(gray):

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    opened = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(opened)

    blob_count = (
        int(num_labels - 1)
        if num_labels > 0 else 0
    )

    areas = [
        int(s[cv2.CC_STAT_AREA])
        for s in stats[1:]
    ] if num_labels > 1 else []

    mean_blob_area = (
        float(np.mean(areas))
        if len(areas) > 0 else 0
    )

    return blob_count, mean_blob_area


def classify_freshness(
    h,
    s,
    contrast,
    homogeneity,
    blob_count
):

    score = 0

    explanation = []

    if (0 <= h <= 40):
        score += 1
        explanation.append(
            "Hue mendekati warna daging segar."
        )

    if s > 40:
        score += 1
        explanation.append(
            "Saturasi warna cukup baik."
        )

    if homogeneity > 0.5:
        score += 1
        explanation.append(
            "Tekstur relatif homogen."
        )

    if contrast < 60:
        score += 1
        explanation.append(
            "Permukaan tidak terlalu kasar."
        )

    if blob_count < 5:
        score += 1
        explanation.append(
            "Jumlah bercak relatif sedikit."
        )

    if score >= 4:
        status = "Segar"
    elif score >= 2:
        status = "Kurang Segar"
    else:
        status = "Tidak Segar"

    return status, explanation
    @app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        file = request.files.get("image")

        if not file:
            error = "Silakan pilih gambar."

        else:

            file_bytes = np.frombuffer(
                file.read(),
                np.uint8
            )

            img = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )

            if img is None:
                error = "File gambar tidak valid."

            else:

                avg_color = img.mean(axis=(0, 1))

                b = int(avg_color[0])
                g = int(avg_color[1])
                r = int(avg_color[2])

                hsv = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2HSV
                )

                avg_hsv = hsv.mean(axis=(0, 1))

                h = float(avg_hsv[0])
                s = float(avg_hsv[1])
                v = float(avg_hsv[2])

                l_lab, a_lab, b_lab = \
                    compute_color_lab(img)

                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2GRAY
                )

                contrast, homogeneity, energy = \
                    compute_glcm_features(gray)

                mean_lbp, uniform_lbp = \
                    compute_lbp_feature(gray)

                gabor_energy = \
                    compute_gabor_features(gray)

                shape = \
                    compute_shape_features(gray)

                blob_count, mean_blob_area = \
                    detect_blobs(gray)

                status, explanation = \
                    classify_freshness(
                        h,
                        s,
                        contrast,
                        homogeneity,
                        blob_count
                    )

                result = {
                    "r": r,
                    "g": g,
                    "b": b,

                    "h": round(h, 2),
                    "s": round(s, 2),
                    "v": round(v, 2),

                    "l_lab": round(l_lab, 2),
                    "a_lab": round(a_lab, 2),
                    "b_lab": round(b_lab, 2),

                    "contrast": round(contrast, 3),
                    "homogeneity": round(homogeneity, 3),
                    "energy": round(energy, 3),

                    "mean_lbp": round(mean_lbp, 3),
                    "uniform_lbp": round(uniform_lbp, 3),

                    "gabor_energy": round(
                        gabor_energy,
                        4
                    ),

                    "area": shape["area"],
                    "perimeter": shape["perimeter"],
                    "aspect_ratio": shape["aspect_ratio"],
                    "convexity": shape["convexity"],
                    "roundness": shape["roundness"],

                    "blob_count": blob_count,
                    "mean_blob_area": round(
                        mean_blob_area,
                        2
                    ),

                    "status": status,
                    "explanation": explanation
                }

    return render_template(
        "index.html",
        result=result,
        error=error
    )

application = app
