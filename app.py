from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Silakan pilih gambar"
        else:
            result = {
                "filename": file.filename,
                "status": "Upload berhasil (Vercel running OK)"
            }

    return render_template("index.html", result=result, error=error)


# WAJIB untuk Vercel
app = app
