import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps

app = Flask(__name__)
modelo = tf.keras.models.load_model("modelo_perros_gatos.keras")
TAM = (160, 160)


@app.route("/")
def inicio():
    return send_from_directory(".", "index.html")


@app.route("/predecir", methods=["POST"])
def predecir():
    if "imagen" not in request.files:
        return jsonify(error="No se envió ninguna imagen"), 400

    img = Image.open(io.BytesIO(request.files["imagen"].read()))
    img = ImageOps.exif_transpose(img).convert("RGB").resize(TAM)  # corrige rotación de fotos del celular
    x = np.expand_dims(np.array(img, dtype="float32"), axis=0)

    prob_perro = float(modelo.predict(x, verbose=0)[0][0])
    etiqueta = "Perro 🐶" if prob_perro >= 0.5 else "Gato 🐱"
    confianza = prob_perro if prob_perro >= 0.5 else 1 - prob_perro

    return jsonify(etiqueta=etiqueta, confianza=round(confianza * 100, 1))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
