from flask import Flask, render_template, request
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Pilih gambar dulu"
        else:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            result = {
                "filename": file.filename,
                "status": "Upload berhasil di Vercel"
            }

    return render_template("index.html", result=result, error=error)


app = app
