from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = "Combined_xception_best.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Ensure the image size matches the model's expected input shape
IMAGE_SIZE = (256, 256)  # Model expects (None, 256, 256, 3)

def preprocess_image(image_path):
    """Load and preprocess an image."""
    img = load_img(image_path, target_size=IMAGE_SIZE)  # Resize to (256, 256)
    img_array = img_to_array(img) / 255.0  # Normalize (0-1 scaling)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    temp_file_path = "temp_image.jpg"
    file.save(temp_file_path)

    try:
        img_array = preprocess_image(temp_file_path)
        prediction = model.predict(img_array)[0][0]  # Extract single value
        os.remove(temp_file_path)  # Clean up temporary file

        # Convert probability into label
        label = "Real" if prediction >= 0.5 else "Fake"
        confidence = round(prediction * 100, 2) if label == "Real" else round((1 - prediction) * 100, 2)

        return jsonify({
            "prediction": label,
            "confidence": f"{confidence}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
