from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
import os
import datetime
from werkzeug.utils import secure_filename  # For secure filenames

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

app = Flask(__name__)

# Load the model
MODEL_PATH = "Combined_xception_best.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Ensure the image size matches the model's expected input shape
IMAGE_SIZE = (256, 256)  # Model expects (None, 256, 256, 3)
UPLOAD_FOLDER = "gradcam_outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure output folder exists

def preprocess_image(image_path):
    """Load and preprocess an image."""
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize (0-1 scaling)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch
    return img_array

def generate_gradcam(image_path, model, last_conv_layer_name="block14_sepconv2_act"):
    """Generate a Grad-CAM heatmap for the given image."""
    
    # Load and preprocess image
    img_array = preprocess_image(image_path)

    # Identify the last convolutional layer
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Grad-CAM for class 0 (Fake)

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature map
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)[0]

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize

    # Convert TensorFlow tensor to NumPy array explicitly
    heatmap = heatmap.numpy() if isinstance(heatmap, tf.Tensor) else heatmap  

    # Load original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # ✅ Fixed

    # Convert heatmap to color
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Save Grad-CAM output
    gradcam_path = os.path.join(UPLOAD_FOLDER, os.path.basename(image_path))
    cv2.imwrite(gradcam_path, superimposed_img)

    return gradcam_path

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    # 1. Secure the original filename (important!)
    original_filename = secure_filename(file.filename)

    # 2. Get current timestamp for uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # YearMonthDayHourMinuteSecond

    # 3. Create the new filename (file name + timestamp + extension)
    name, ext = os.path.splitext(original_filename) # Split name and extension
    new_filename = f"{name}_{timestamp}{ext}"  # or name + "_" + timestamp + ext

    temp_file_path = os.path.join(UPLOAD_FOLDER, new_filename)
    file.save(temp_file_path)

    try:
        img_array = preprocess_image(temp_file_path)
        prediction = model.predict(img_array)[0][0]  # Extract single value

        # Determine Real/Fake
        label = "Real" if prediction >= 0.5 else "Fake"
        confidence = round(prediction * 100, 2) if label == "Real" else round((1 - prediction) * 100, 2)

        # Generate Grad-CAM
        gradcam_path = generate_gradcam(temp_file_path, model)

        return jsonify({
            "prediction": label,
            "confidence": f"{confidence}%",
            "gradcam_url": f"http://127.0.0.1:5000/gradcam/{os.path.basename(gradcam_path)}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/gradcam/<filename>', methods=['GET'])
def serve_gradcam(filename):
    """Serve the Grad-CAM image."""
    return send_from_directory(UPLOAD_FOLDER, filename)

###########=>    VIDEOS ###
# Constants
FRAME_EXTRACT_RATE = 10   # Extract every 10th frame to reduce processing time
fake_frames_dir = "fake_frames"
os.makedirs(fake_frames_dir, exist_ok=True)

def preprocess_frame(frame):
    """Preprocess a single video frame for model prediction."""
    frame = cv2.resize(frame, IMAGE_SIZE)  # Resize to model's input size
    frame = img_to_array(frame) / 255.0    # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Expand dimensions for batch
    return frame

def analyze_video(video_path):
    """Extract frames from video and classify each frame."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_count = 0
    real_count = 0
    frame_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        if frame_count % FRAME_EXTRACT_RATE == 0:  # Process every Nth frame
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)[0][0]  # Get probability
            
            label = "Real" if prediction >= 0.5 else "Fake"
            frame_results.append({"frame": frame_count, "prediction": label})

            if label == "Fake":
                fake_count += 1
                # Save the fake frame
                fake_frame_filename = os.path.join(fake_frames_dir, f"frame_{frame_count}.jpg")  # or .png
                cv2.imwrite(fake_frame_filename, frame)
            else:
                real_count += 1

        frame_count += 1

    cap.release()

    # Calculate percentages
    total_analyzed = fake_count + real_count
    fake_percentage = round((fake_count / total_analyzed) * 100, 2) if total_analyzed > 0 else 0
    real_percentage = round((real_count / total_analyzed) * 100, 2) if total_analyzed > 0 else 0

    return {
        "total_frames_analyzed": total_analyzed,
        "fake_percentage": fake_percentage,
        "real_percentage": real_percentage,
        "sample_results": frame_results[:5]  # Return first 5 frame results for debugging
    }

@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['file']
    temp_video_path = os.path.join(UPLOAD_FOLDER, "input_video.mp4")
    file.save(temp_video_path)

    try:
        results = analyze_video(temp_video_path)
        os.remove(temp_video_path)  # Cleanup uploaded file
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
