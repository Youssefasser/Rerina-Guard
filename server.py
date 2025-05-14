from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
model = load_model("2025.h5")

class_labels = ["CNV", "DME", "Drusen", "Normal"]

def preprocess_image(img_bytes):
    img_array = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image = preprocess_image(file.read())

    prediction = model.predict(image)[0]
    predicted_label = class_labels[np.argmax(prediction)]

    return jsonify({ "predicted_class": predicted_label })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
