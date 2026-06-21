from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import cv2
from skimage.feature import local_binary_pattern

app = Flask(__name__)

def analyze(image):
    img = np.array(image)
    img = cv2.resize(img, (200, 200))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    red_mean = np.mean(hsv[:, :, 0])

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    texture = np.mean(lbp)

    if red_mean < 10 and texture < 5:
        return "TIDAK SEGAR 🔴"
    elif red_mean < 20:
        return "KURANG SEGAR 🟡"
    else:
        return "SEGAR 🟢"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    result = analyze(image)

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run()
