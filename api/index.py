from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# WAJIB untuk Vercel (WSGI handler)
def handler(environ, start_response):
    return app(environ, start_response)
