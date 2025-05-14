from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2


app = Flask(__name__)
# Load the pre-trained model
# Ensure the model file is in the same directory or provide the correct path

model = load_model("retina_guard_cnn_v1_acc99_4.h5")
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
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')  # Accept multiple files under 'images'

    if len(files) != 2:
        return jsonify({'error': 'Exactly 2 images are required'}), 400

    predictions = []

    for file in files:
        image = preprocess_image(file.read())
        prediction = model.predict(image)[0]
        predicted_label = class_labels[np.argmax(prediction)]
        predictions.append(predicted_label)

    return jsonify({
        "prediction_image1": predictions[0],
        "prediction_image2": predictions[1]
    })
@app.route('/')
def index():
    return "Welcome to the Retina Guard API! Use /predict to get predictions."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
# To run the server, use the command:
# python server.py