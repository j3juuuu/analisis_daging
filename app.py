from flask import Flask, render_template, request
import cv2
import numpy as np

app = Flask(__name__)

# =========================
# SIMPLE ANALYSIS FUNCTION
# =========================
def analyze_meat(gray):
    mean_val = np.mean(gray)

    if mean_val > 120:
        status = "Segar"
    elif mean_val > 70:
        status = "Kurang Segar"
    else:
        status = "Tidak Segar"

    return status


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
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                status = analyze_meat(gray)

                result = {
                    "status": status,
                    "mean": float(np.mean(gray))
                }

    return render_template("index.html", result=result, error=error)


# =========================
# VERCEL ENTRY
# =========================
app = app
