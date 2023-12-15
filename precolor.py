import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load the TensorFlow model
model_path = "model_color_recognition_DSV3.h5"
model = tf.keras.models.load_model(model_path)

color_labels = ['abuabu', 'biru', 'coklat', 'hijau', 'hitam', 'kuning', 'merah', 'orange', 'pink', 'putih', 'silver', 'ungu']


def transform_image(pillow_image):
    if pillow_image.mode == 'RGBA':
        pillow_image = pillow_image.convert('RGB')
    data = np.asarray(pillow_image)
    data = data.astype(np.float32)  # Convert to FLOAT32
    data /= 255.0  # Normalize to the range [0, 1]
    data = np.expand_dims(data, axis=0)
    return data

def predict_color(x):
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions)
    predicted_color_label = color_labels[predicted_class]
    max_prob = np.max(predictions)
    return predicted_class, predicted_color_label, max_prob

@app.route("/", methods=["GET","POST"])
def ismprecolor():
    try:
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        # Read and process the image using PIL.Image
        pillow_img = Image.open(io.BytesIO(file.read()))
        pillow_img = pillow_img.resize((224, 224))
        img_array = transform_image(pillow_img)

        # Perform color prediction
        predicted_class, predicted_color_label, max_prob = predict_color(img_array)

        # Create a dictionary for the output
        output_dict = {
            "fitur": "Prediksi Warna Object",
            "gambar": file.filename,
            "prediksi_kelas": int(predicted_class),
            "prediksi_warna": predicted_color_label,
            "prediksi_akurasi": f"{round(max_prob * 100, 2)}%"
        }

        return jsonify(output_dict)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
