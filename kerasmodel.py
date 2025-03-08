from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
import os
import datetime
from werkzeug.utils import secure_filename  # For secure filenames
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import concurrent.futures
from queue import Queue
import time
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

#MODEL_PATH = "Combined_xception_best.keras"
MODEL_PATH = "FinalDF_xception_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

IMAGE_SIZE = (256, 256)  # Model expects (None, 256, 256, 3)
UPLOAD_FOLDER = "gradcam_outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  


def generate_gradcam(image, model, last_conv_layer_name="block14_sepconv2_act"):
    """Generate a Grad-CAM heatmap for a given image (without using a file path)."""
    
    # Preprocess image (ensure it's correctly formatted for model input)
    img_array = preprocess_pil_image(image)

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

    # Convert PIL Image to NumPy array for heatmap overlay
    img = np.array(image)  # Convert PIL image to NumPy array
    img = cv2.resize(img, IMAGE_SIZE)  # Ensure it's the correct size

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to color
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Save Grad-CAM output
    gradcam_filename = f"gradcam_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
    cv2.imwrite(gradcam_path, superimposed_img)

    return gradcam_path  # Return the saved file path

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    original_filename = secure_filename(file.filename)

    # Open the image as a PIL Image
    image = Image.open(file.stream).convert("RGB")  # Convert to RGB to ensure correct format

    try:
        # Preprocess image
        img_array = preprocess_pil_image(image)
        prediction = model.predict(img_array)[0][0]  # Extract single value

        # Determine Real/Fake
        label = "Real" if prediction >= 0.5 else "Fake"
        confidence = round(prediction * 100, 2) if label == "Real" else round((1 - prediction) * 100, 2)

        # Generate Grad-CAM using in-memory image
        gradcam_path = generate_gradcam(image, model)

        return jsonify({
            "prediction": label,
            "confidence": f"{confidence}%",
            "gradcam_url": f"http://127.0.0.1:5000/gradcam/{os.path.basename(gradcam_path)}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


from PIL import Image
from io import BytesIO
import requests

@app.route('/predict_url', methods=['POST'])
def predict_url():
    # Get image URL from JSON request
    data = request.get_json()
    
    # Check if image URL is provided
    if 'image_url' not in data:
        return jsonify({"error": "No image URL provided"}), 400
    
    image_url = data['image_url']
    
    try:
        # Download the image from the URL
        response = requests.get(image_url)
        
        # Check if the request was successful
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 400
        
        # Open the image from the downloaded content
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Preprocess image
        img_array = preprocess_pil_image(image)
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]  # Extract single value
        
        # Determine Real/Fake
        label = "Real" if prediction >= 0.5 else "Fake"
        confidence = round(prediction * 100, 2) if label == "Real" else round((1 - prediction) * 100, 2)
        
        # Generate Grad-CAM using in-memory image
        gradcam_path = generate_gradcam(image, model)
        
        return jsonify({
            "prediction": label,
            "confidence": f"{confidence}%",
            "gradcam_url": f"http://127.0.0.1:5000/gradcam/{os.path.basename(gradcam_path)}"
        })
    
    except requests.RequestException as e:
        return jsonify({"error": f"Network error: {str(e)}"}), 500
    
    except Image.UnidentifiedImageError:
        return jsonify({"error": "Invalid image format"}), 400
    
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



def frame_to_image(frame):
    """Convert an OpenCV video frame to a PIL image (without saving it)."""
    # Convert from BGR (OpenCV format) to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image (this mimics reading an image file)
    image = Image.fromarray(frame_rgb)
    
    return image
def preprocess_pil_image(image):
    """Preprocess a PIL Image to match model input."""
    image = image.resize(IMAGE_SIZE)  # Resize to (256, 256)
    img_array = img_to_array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def process_frame(frame, frame_count):
    """Process a single frame."""
    image = frame_to_image(frame)
    processed_frame = preprocess_pil_image(image)
    prediction = model.predict(processed_frame)[0][0]  # Get probability
    label = "Real" if prediction >= 0.5 else "Fake"
    
    if label == "Fake":
        fake_frame_filename = os.path.join(fake_frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(fake_frame_filename, frame)
    
    return frame_count, label

def analyze_video(video_path):
    """Extract frames from video and classify each frame using multithreading."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_count = 0
    real_count = 0
    frame_results = []
    
    frame_queue = Queue()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends
        
        if frame_count % FRAME_EXTRACT_RATE == 0:  # Process every Nth frame
            frame_queue.put((frame, frame_count))
        
        frame_count += 1
    
    cap.release()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_frame, frame, count): (frame, count) for frame, count in list(frame_queue.queue)}
        
        for future in concurrent.futures.as_completed(futures):
            frame_number, label = future.result()
            frame_results.append({"frame": frame_number, "prediction": label})
            if label == "Fake":
                fake_count += 1
            else:
                real_count += 1
    
    total_analyzed = fake_count + real_count
    fake_percentage = round((fake_count / total_analyzed) * 100, 2) if total_analyzed > 0 else 0
    real_percentage = round((real_count / total_analyzed) * 100, 2) if total_analyzed > 0 else 0
    
    return {
        "total_frames_analyzed": total_analyzed,
        "fake_percentage": fake_percentage,
        "real_percentage": real_percentage,
    }


@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['file']
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    original_filename = file.filename
    new_filename = f"{current_time}_{original_filename}"
    temp_video_path = os.path.join(UPLOAD_FOLDER, new_filename)

    file.save(temp_video_path)

    try:
        start_time = time.time()
        results = analyze_video(temp_video_path)
        os.remove(temp_video_path)  # Cleanup uploaded file
        end_time = time.time() 
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
