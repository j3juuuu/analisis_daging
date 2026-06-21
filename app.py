from flask import Flask, render_template, request
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# pastikan folder ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Silakan pilih gambar dulu."
        else:
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # versi sederhana (tanpa cv2/scipy biar tidak crash)
            result = {
                "filename": filename,
                "status": "Gambar berhasil diupload (versi demo aman Vercel)"
            }

    return render_template("index.html", result=result, error=error)


# penting untuk Vercel
application = app
