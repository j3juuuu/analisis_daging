from flask import Flask, render_template, request
import cv2
import numpy as np
import math
import base64
from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

app = Flask(__name__)

# =========================
# GLCM FEATURE
# =========================

def compute_glcm_features(gray, levels=16):
    image = np.floor(gray.astype(np.float64) / 256 * levels).astype(np.uint8)
    h, w = image.shape

    glcm = np.zeros((levels, levels), dtype=np.float64)

    offsets = [(1,0),(0,1),(1,1),(-1,1)]

    for dy, dx in offsets:
        for y in range(h):
            for x in range(w):
                ny, nx = y + dy, x + dx

                if 0 <= ny < h and 0 <= nx < w:
                    glcm[image[y,x], image[ny,nx]] += 1

    if glcm.sum() == 0:
        return 0.0,0.0,0.0

    glcm /= glcm.sum()

    contrast = 0
    homogeneity = 0
    energy = 0

    for i in range(levels):
        for j in range(levels):
            pij = glcm[i,j]

            contrast += pij*((i-j)**2)
            homogeneity += pij/(1+abs(i-j))
            energy += pij*pij

    return contrast, homogeneity, energy

# =========================
# LBP FEATURE
# =========================

def compute_lbp_feature(gray):
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')

    mean_lbp = float(lbp.mean())

    hist,_ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0,11),
        density=True
    )

    uniform_prop = float(hist[0]) if len(hist)>0 else 0

    return mean_lbp, uniform_prop

# =========================
# GABOR FEATURE
# =========================

def compute_gabor_features(gray):

    gray_f = gray.astype(np.float32)/255.0

    energies = []

    frequencies = [0.1,0.2]
    thetas = [0,np.pi/4,np.pi/2,3*np.pi/4]

    for freq in frequencies:
        for theta in thetas:

            kernel = np.real(
                gabor_kernel(freq, theta=theta)
            )

            filtered = ndi.convolve(
                gray_f,
                kernel,
                mode='reflect'
            )

            energies.append(
                float(np.mean(np.abs(filtered)))
            )

    return float(np.mean(energies))

# =========================
# COLOR LAB
# =========================

def compute_color_lab(img):

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    avg = lab.mean(axis=(0,1))

    return float(avg[0]),float(avg[1]),float(avg[2])

# =========================
# SHAPE FEATURE
# =========================

def compute_shape_features(gray):

    blur = cv2.GaussianBlur(gray,(5,5),0)

    _,thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {
            "area":0,
            "perimeter":0,
            "compactness":0
        }

    contour = max(contours,key=cv2.contourArea)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)

    compactness = 0

    if perimeter > 0:
        compactness = (
            4*math.pi*area
        )/(perimeter*perimeter)

    return {
        "area":int(area),
        "perimeter":int(perimeter),
        "compactness":round(compactness,3)
    }

# =========================
# CLASSIFICATION
# =========================

def classify_freshness(color, texture):

    r,g,b,h,s,v = color

    contrast,homogeneity,energy = texture

    score = 0

    explanation = []

    if s > 40:
        score += 1
        explanation.append(
            "Warna masih cukup segar."
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

    if score >= 3:
        status = "Segar"
    elif score == 2:
        status = "Kurang Segar"
    else:
        status = "Tidak Segar"

    return status, explanation

@app.route("/", methods=["GET","POST"])
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
                error = "Gambar tidak valid."

            else:

                avg = img.mean(axis=(0,1))

                b = int(avg[0])
                g = int(avg[1])
                r = int(avg[2])

                hsv = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2HSV
                )

                avg_hsv = hsv.mean(axis=(0,1))

                h = float(avg_hsv[0])
                s = float(avg_hsv[1])
                v = float(avg_hsv[2])

                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2GRAY
                )

                contrast,homogeneity,energy = \
                    compute_glcm_features(gray)

                mean_lbp,uniform_prop = \
                    compute_lbp_feature(gray)

                gabor_energy = \
                    compute_gabor_features(gray)

                shape = \
                    compute_shape_features(gray)

                l_lab,a_lab,b_lab = \
                    compute_color_lab(img)

                status,explanation = \
                    classify_freshness(
                        (r,g,b,h,s,v),
                        (contrast,homogeneity,energy)
                    )

                # Additional shape features
                bbox = "N/A"
                aspect_ratio = 1.0
                convexity = 1.0
                roundness = 1.0
                blob_count = 1
                mean_blob_area = shape["area"]

                # Encode image to base64
                _, img_encoded = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(img_encoded).decode('utf-8')

                result = {
                    "filename":file.filename,
                    "image_base64":"data:image/jpeg;base64," + img_base64,
                    "r":r,
                    "g":g,
                    "b":b,
                    "h":round(h,1),
                    "s":round(s,1),
                    "v":round(v,1),
                    "contrast":round(contrast,2),
                    "homogeneity":round(homogeneity,3),
                    "energy":round(energy,3),
                    "l_lab":round(l_lab,1),
                    "a_lab":round(a_lab,1),
                    "b_lab":round(b_lab,1),
                    "mean_lbp":round(mean_lbp,3),
                    "uniform_lbp":round(uniform_prop,3),
                    "gabor_energy":round(gabor_energy,4),
                    "area":shape["area"],
                    "perimeter":shape["perimeter"],
                    "compactness":shape["compactness"],
                    "bbox":bbox,
                    "aspect_ratio":round(aspect_ratio,3),
                    "convexity":round(convexity,3),
                    "roundness":round(roundness,3),
                    "blob_count":blob_count,
                    "mean_blob_area":mean_blob_area,
                    "status":status,
                    "explanation":explanation
                }

    return render_template(
        "index.html",
        result=result,
        error=error
    )

application = app

if __name__ == "__main__":
    app.run(debug=True)
