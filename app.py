from flask import Flask, request, jsonify, render_template

from PIL import Image
import PIL
from prediction import word_predictor
import os
import cv2
import numpy as np

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def upload_picture():

    file = request.files["image"]
    img = Image.open(file.stream)
    img.save("static/image.jpg")
    return render_template("index.html", success="Image Posted sucessfully")


@app.route("/predict", methods=["GET"])
def prediction():
    img = cv2.imread("static/image.jpg")
    result = word_predictor(img)
    # os.remove('staic/image.jpg')
    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0")
