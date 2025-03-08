from flask import Flask, request, jsonify, send_from_directory, send_file
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import os
import datetime
import time
import zipfile
import shutil
from werkzeug.utils import secure_filename
from PIL import Image
import concurrent.futures
from queue import Queue
from flask_cors import CORS
import requests
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

app = Flask(__name__)
CORS(app)

# Common settings
IMAGE_SIZE = (256, 256)
UPLOAD_FOLDER = "gradcam_outputs"
FRAME_EXTRACT_RATE = 10

# Add new directories for organizing fake frames
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
fake_frames_dir = "fake_frames"
os.makedirs(fake_frames_dir, exist_ok=True)
tf_fake_frames_dir = os.path.join(fake_frames_dir, "tensorflow")
torch_fake_frames_dir = os.path.join(fake_frames_dir, "pytorch")
common_fake_frames_dir = os.path.join(fake_frames_dir, "common")
zip_output_dir = os.path.join(UPLOAD_FOLDER, "zip_files")
os.makedirs(tf_fake_frames_dir, exist_ok=True)
os.makedirs(torch_fake_frames_dir, exist_ok=True)
os.makedirs(common_fake_frames_dir, exist_ok=True)
os.makedirs(zip_output_dir, exist_ok=True)

# Models paths (assuming these are defined in your original code)
TENSORFLOW_MODEL_PATH = "FinalDF_xception_model.keras"
PYTORCH_MODEL_PATH = "best_swin_model.pth"

# Load TensorFlow model
tf_model = tf.keras.models.load_model(TENSORFLOW_MODEL_PATH)

# Define the Swin Transformer model class
class SwinTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super(SwinTransformer, self).__init__()
        self.model = models.swin_t(weights='DEFAULT')
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# Load the PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_model = SwinTransformer(num_classes=2)
torch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
torch_model.to(device)
torch_model.eval()

# Define preprocessing transform
preprocess_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def tf_predict_image(image):
    """Predict if an image is real or fake using the TensorFlow model."""
    # Preprocess image
    img_array = tf_preprocess_pil_image(image)
    prediction = tf_model.predict(img_array)[0][0]  # Extract single value

    # Determine Real/Fake
    label = "Real" if prediction >= 0.5 else "Fake"
    confidence = round(prediction * 100, 2) if label == "Real" else round((1 - prediction) * 100, 2)

    # Generate Grad-CAM using in-memory image
    gradcam_path = tf_generate_gradcam(image, tf_model)

    return {
        "prediction": label,
        "confidence": f"{confidence}%",
        "gradcam_url": f"http://127.0.0.1:5000/gradcam/{os.path.basename(gradcam_path)}",
        "Model": "Xception"
    }
def tf_generate_gradcam(image, model, last_conv_layer_name="block14_sepconv2_act"):
    """Generate a Grad-CAM heatmap using TensorFlow model."""
    
    # Preprocess image
    img_array = tf_preprocess_pil_image(image)

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
    heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1.0  # Normalize

    # Convert TensorFlow tensor to NumPy array explicitly
    heatmap = heatmap.numpy() if isinstance(heatmap, tf.Tensor) else heatmap  

    # Convert PIL Image to NumPy array for heatmap overlay
    img = np.array(image)
    img = cv2.resize(img, IMAGE_SIZE)

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to color
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Save Grad-CAM output
    gradcam_filename = f"gradcam_tf_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
    cv2.imwrite(gradcam_path, superimposed_img)

    return gradcam_path

def torch_preprocess_pil_image(image):
    """Preprocess a PIL Image for PyTorch model."""
    img_tensor = preprocess_transform(image).unsqueeze(0).to(device)
    return img_tensor

def torch_generate_gradcam(image, model, target_layer_name="model.features.7"):
    """Generate a Grad-CAM heatmap using PyTorch model."""
    
    # Preprocess image
    img_tensor = torch_preprocess_pil_image(image)
    
    # Get the target layer
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    else:
        # If the target layer isn't found, try an alternative layer
        target_layer = None
        for name, module in model.named_modules():
            if "norm" in name and hasattr(module, "weight"):
                target_layer = module
                break
        
        if target_layer is None:
            # Fallback if no suitable layer is found
            img_array = np.array(image.resize(IMAGE_SIZE))
            gradcam_filename = f"gradcam_torch_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
            cv2.imwrite(gradcam_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            return gradcam_path
    
    # Register hooks
    activations = []
    gradients = []
    
    def save_activation(module, input, output):
        activations.append(output)
    
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register forward and backward hooks
    handle1 = target_layer.register_forward_hook(save_activation)
    handle2 = target_layer.register_full_backward_hook(save_gradient)
    
    # Forward pass
    model.zero_grad()
    output = model(img_tensor)
    
    # Get the score for the fake class (index 1)
    pred_scores = torch.softmax(output, dim=1)
    score = pred_scores[0, 1]  # Get score for fake class
    
    # Backward pass
    score.backward()
    
    # Remove hooks
    handle1.remove()
    handle2.remove()
    
    # Get the gradient and activation
    if len(gradients) == 0 or len(activations) == 0:
        # Fallback if GradCAM fails
        img_array = np.array(image.resize(IMAGE_SIZE))
        gradcam_filename = f"gradcam_torch_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        return gradcam_path
    
    try:
        # For Swin Transformer, the output may need reshaping
        activation = activations[0]
        gradient = gradients[0]
        
        # Handle different possible shapes for Swin Transformer
        if len(activation.shape) == 4:  # If already in [batch, channels, height, width] format
            pass  # Keep as is
        elif len(activation.shape) == 3:  # If in [batch, tokens, channels] format
            b, n, c = activation.shape
            h = w = int(np.sqrt(n))
            activation = activation.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [batch, channels, height, width]
        else:
            # For other shapes, use a fallback method
            img_array = np.array(image.resize(IMAGE_SIZE))
            gradcam_filename = f"gradcam_torch_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
            cv2.imwrite(gradcam_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            return gradcam_path
        
        # Handle gradient shape similarly
        if len(gradient.shape) == 4:  # If already in [batch, channels, height, width] format
            pass  # Keep as is
        elif len(gradient.shape) == 3:  # If in [batch, tokens, channels] format
            b, n, c = gradient.shape
            h = w = int(np.sqrt(n))
            gradient = gradient.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [batch, channels, height, width]
        
        # Calculate importance weights
        if len(gradient.shape) == 4:
            weights = torch.mean(gradient[0], dim=(1, 2))
        else:
            weights = torch.mean(gradient.view(gradient.size(0), -1), dim=1)
        
        # Create weighted activation map
        if len(activation.shape) == 4:
            cam = torch.zeros(activation.shape[2:], dtype=torch.float32, device=device)
            for i, w in enumerate(weights):
                cam += w * activation[0, i]
        else:
            cam = torch.sum(weights.view(-1, 1, 1) * activation[0], dim=0)
            
        # Apply ReLU and normalize
        cam = torch.maximum(cam, torch.tensor(0., device=device))
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        # Convert to numpy and resize
        cam = cam.cpu().detach().numpy()
        
        # Convert PIL Image to NumPy array for heatmap overlay
        img = np.array(image)
        img = cv2.resize(img, IMAGE_SIZE)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to color
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert to BGR if needed for OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        
        # Overlay heatmap on original image
        superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
        
        # Save Grad-CAM output
        gradcam_filename = f"gradcam_torch_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
        cv2.imwrite(gradcam_path, superimposed_img)
        
        return gradcam_path
        
    except Exception as e:
        print(f"Error in PyTorch GradCAM generation: {str(e)}")
        # Fallback - return original image with timestamp if GradCAM fails
        img_array = np.array(image.resize(IMAGE_SIZE))
        gradcam_filename = f"gradcam_torch_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        return gradcam_path

def torch_predict_image(image):
    """Predict if an image is real or fake using the PyTorch model."""
    # Preprocess image
    img_tensor = torch_preprocess_pil_image(image)
    
    # Get prediction
    with torch.no_grad():
        output = torch_model(img_tensor)
        probs = torch.softmax(output, dim=1)
        prediction = probs[0][0].item()  # Index 0 for Real probability
    
    # Determine Real/Fake
    label = "Real" if prediction >= 0.5 else "Fake"
    confidence = round(prediction * 100, 2) if label == "Real" else round((1 - prediction) * 100, 2)

    # Generate Grad-CAM using in-memory image
    gradcam_path = torch_generate_gradcam(image, torch_model)

    return {
        "prediction": label,
        "confidence": f"{confidence}%",
        "gradcam_url": f"http://127.0.0.1:5000/gradcam/{os.path.basename(gradcam_path)}",
        "Model": "swin"
    }

# ============================ COMBINED ENDPOINTS ============================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    # Open the image as a PIL Image
    image = Image.open(file.stream).convert("RGB")
    
    try:
        # Get predictions from both models
        tf_result = tf_predict_image(image)
        torch_result = torch_predict_image(image)
        
        # Return results as a list
        return jsonify([tf_result, torch_result])
    except Exception as e:
        print(f"Error in /predict endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
        
        # Get predictions from both models
        tf_result = tf_predict_image(image)
        torch_result = torch_predict_image(image)
        
        # Return results as a list
        return jsonify([tf_result, torch_result])
    
    except requests.RequestException as e:
        return jsonify({"error": f"Network error: {str(e)}"}), 500
    
    except Image.UnidentifiedImageError:
        return jsonify({"error": "Invalid image format"}), 400
    
    except Exception as e:
        print(f"Error in /predict_url endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/gradcam/<filename>', methods=['GET'])
def serve_gradcam(filename):
    """Serve the Grad-CAM image."""
    return send_from_directory(UPLOAD_FOLDER, filename)

# Define preprocessing transform for PyTorch
preprocess_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def tf_preprocess_pil_image(image):
    """Preprocess a PIL Image for TensorFlow model."""
    image = image.resize(IMAGE_SIZE)
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def torch_preprocess_pil_image(image):
    """Preprocess a PIL Image for PyTorch model."""
    img_tensor = preprocess_transform(image).unsqueeze(0).to(device)
    return img_tensor

def frame_to_image(frame):
    """Convert an OpenCV video frame to a PIL image."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def tf_process_frame(frame, frame_count):
    """Process a single frame with TensorFlow model."""
    image = frame_to_image(frame)
    processed_frame = tf_preprocess_pil_image(image)
    prediction = tf_model.predict(processed_frame)[0][0]
    label = "Real" if prediction >= 0.5 else "Fake"
    is_fake = label == "Fake"
    
    if is_fake:
        fake_frame_filename = os.path.join(tf_fake_frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(fake_frame_filename, frame)
    
    return frame_count, label, is_fake

def torch_process_frame(frame, frame_count):
    """Process a single frame with PyTorch model."""
    image = frame_to_image(frame)
    img_tensor = torch_preprocess_pil_image(image)
    
    with torch.no_grad():
        output = torch_model(img_tensor)
        probs = torch.softmax(output, dim=1)
        prediction = probs[0][0].item()
    
    label = "Real" if prediction >= 0.5 else "Fake"
    is_fake = label == "Fake"
    
    if is_fake:
        fake_frame_filename = os.path.join(torch_fake_frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(fake_frame_filename, frame)
    
    return frame_count, label, is_fake

def create_zip_file(source_dir, zip_filename):
    """Create a ZIP file from a directory."""
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
                zipf.write(file_path, arcname)
    return zip_filename

def find_common_fake_frames(tf_results, torch_results):
    """Find frame indices that both models identified as fake."""
    tf_fake_frames = set(frame_count for frame_count, _, is_fake in tf_results if is_fake)
    torch_fake_frames = set(frame_count for frame_count, _, is_fake in torch_results if is_fake)
    
    # Get common frame indices
    common_frames = tf_fake_frames.intersection(torch_fake_frames)
    return common_frames

def analyze_video(video_path):
    """Analyze a video with both models and create ZIP files of fake frames."""
    # Clear previous frame directories
    for directory in [tf_fake_frames_dir, torch_fake_frames_dir, common_fake_frames_dir]:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    # Create a unique session ID for this video analysis
    session_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    tf_fake_count = 0
    tf_real_count = 0
    torch_fake_count = 0
    torch_real_count = 0
    
    frames = []
    
    # Extract frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % FRAME_EXTRACT_RATE == 0:
            frames.append((frame, frame_count))
        
        frame_count += 1
    
    cap.release()
    
    tf_results = []
    torch_results = []
    
    # Process frames with both models
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # TensorFlow model processing
        tf_futures = {executor.submit(tf_process_frame, frame, count): (frame, count) 
                    for frame, count in frames}
        
        # Process TensorFlow results
        for future in concurrent.futures.as_completed(tf_futures):
            result = future.result()
            tf_results.append(result)
            frame_count, label, is_fake = result
            if is_fake:
                tf_fake_count += 1
            else:
                tf_real_count += 1
        
        # PyTorch model processing
        torch_futures = {executor.submit(torch_process_frame, frame, count): (frame, count) 
                       for frame, count in frames}
        
        # Process PyTorch results
        for future in concurrent.futures.as_completed(torch_futures):
            result = future.result()
            torch_results.append(result)
            frame_count, label, is_fake = result
            if is_fake:
                torch_fake_count += 1
            else:
                torch_real_count += 1
    
    # Find common fake frames
    common_frames = find_common_fake_frames(tf_results, torch_results)
    
    # Copy common fake frames to the common directory
    for frame_idx in common_frames:
        original_frame_path = os.path.join(tf_fake_frames_dir, f"frame_{frame_idx:04d}.jpg")
        if os.path.exists(original_frame_path):
            common_frame_path = os.path.join(common_fake_frames_dir, f"frame_{frame_idx:04d}.jpg")
            shutil.copy2(original_frame_path, common_frame_path)
    
    # Create ZIP files
    tf_zip_path = os.path.join(zip_output_dir, f"tensorflow_fake_frames_{session_id}.zip")
    torch_zip_path = os.path.join(zip_output_dir, f"pytorch_fake_frames_{session_id}.zip")
    common_zip_path = os.path.join(zip_output_dir, f"common_fake_frames_{session_id}.zip")
    
    tf_zip_file = create_zip_file(tf_fake_frames_dir, tf_zip_path)
    torch_zip_file = create_zip_file(torch_fake_frames_dir, torch_zip_path)
    common_zip_file = create_zip_file(common_fake_frames_dir, common_zip_path)
    
    # Calculate statistics for TensorFlow model
    tf_total = tf_fake_count + tf_real_count
    tf_fake_percentage = round((tf_fake_count / tf_total) * 100, 2) if tf_total > 0 else 0
    tf_real_percentage = round((tf_real_count / tf_total) * 100, 2) if tf_total > 0 else 0
    
    # Calculate statistics for PyTorch model
    torch_total = torch_fake_count + torch_real_count
    torch_fake_percentage = round((torch_fake_count / torch_total) * 100, 2) if torch_total > 0 else 0
    torch_real_percentage = round((torch_real_count / torch_total) * 100, 2) if torch_total > 0 else 0
    
    # Return combined results with download links
    return [
        {
            "total_frames_analyzed": tf_total,
            "fake_frames_count": tf_fake_count,
            "fake_percentage": tf_fake_percentage,
            "real_percentage": tf_real_percentage,
            "fake_frames_zip": f"http://127.0.0.1:5000/download/{os.path.basename(tf_zip_file)}",
            "Model": "Xception"
        },
        {
            "total_frames_analyzed": torch_total,
            "fake_frames_count": torch_fake_count,
            "fake_percentage": torch_fake_percentage,
            "real_percentage": torch_real_percentage,
            "fake_frames_zip": f"http://127.0.0.1:5000/download/{os.path.basename(torch_zip_file)}",
            "Model": "swin"
        },
        {
            "common_fake_frames_count": len(common_frames),
            "common_fake_frames_zip": f"http://127.0.0.1:5000/download/{os.path.basename(common_zip_file)}",
            "Model": "Common"
        }
    ], session_id

@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['file']
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    original_filename = secure_filename(file.filename) if file.filename else f"video_{current_time}.mp4"
    new_filename = f"{current_time}_{original_filename}"
    temp_video_path = os.path.join(UPLOAD_FOLDER, new_filename)

    file.save(temp_video_path)

    try:
        start_time = time.time()
        results, session_id = analyze_video(temp_video_path)
      
        
        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"Video analysis completed in {elapsed_time:.2f} seconds")
        
        # Clean up the uploaded video file
        os.remove(temp_video_path)
        
        return jsonify(
             results)
        
    except Exception as e:
        print(f"Error in /predict_video endpoint: {str(e)}")
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)  # Cleanup on error
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download a ZIP file."""
    return send_from_directory(zip_output_dir, filename, as_attachment=True)

@app.route('/video/sessions', methods=['GET'])
def get_video_sessions():
    """Get a list of all video analysis sessions."""
    sessions = []
    
    for filename in os.listdir(zip_output_dir):
        if filename.endswith('.zip'):
            parts = filename.split('_')
            if len(parts) >= 3:
                model_type = parts[0]
                session_id = parts[-1].replace('.zip', '')
                
                # Check if this session is already in our list
                session_found = False
                for session in sessions:
                    if session['session_id'] == session_id:
                        session_found = True
                        break
                
                if not session_found:
                    sessions.append({
                        'session_id': session_id,
                        'timestamp': session_id  # Format: YYYYMMDDHHmmss
                    })
    
    return jsonify(sessions)

@app.route('/video/session/<session_id>', methods=['GET'])
def get_session_details(session_id):
    """Get details for a specific video analysis session."""
    session_files = []
    
    for filename in os.listdir(zip_output_dir):
        if session_id in filename and filename.endswith('.zip'):
            model_type = "common" if "common" in filename else "tensorflow" if "tensorflow" in filename else "pytorch"
            session_files.append({
                'filename': filename,
                'model': model_type,
                'download_url': f"http://127.0.0.1:5000/download/{filename}"
            })
    
    if not session_files:
        return jsonify({"error": "Session not found"}), 404
    
    return jsonify({
        'session_id': session_id,
        'timestamp': session_id,  # Format: YYYYMMDDHHmmss
        'files': session_files
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)