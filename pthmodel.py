from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import os
import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import concurrent.futures
from queue import Queue
import time
from flask_cors import CORS
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)

MODEL_PATH = "best_swin_model.pth"
IMAGE_SIZE = (256, 256)
UPLOAD_FOLDER = "gradcam_outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
model = SwinTransformer(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define preprocessing transform
preprocess_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_pil_image(image):
    """Preprocess a PIL Image to match model input for PyTorch."""
    img_tensor = preprocess_transform(image).unsqueeze(0).to(device)
    return img_tensor

def generate_gradcam(image, model, target_layer_name="model.features.7"):
    """Generate a Grad-CAM heatmap for a given image using PyTorch."""
    
    # Preprocess image
    img_tensor = preprocess_pil_image(image)
    
    # Get the target layer - for Swin Transformer, we target the last stage
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    else:
        # If the target layer isn't found, try an alternative layer
        target_layer = None
        for name, module in model.named_modules():
            if "norm" in name and hasattr(module, "weight"):  # Look for a normalization layer
                target_layer = module
                print(f"Using alternative layer: {name}")
                break
        
        if target_layer is None:
            # Fallback if no suitable layer is found
            img_array = np.array(image.resize(IMAGE_SIZE))
            gradcam_filename = f"gradcam_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
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
    
    # Get the score for the fake class (index 1 now, since 0 is real and 1 is fake)
    pred_scores = torch.softmax(output, dim=1)
    score = pred_scores[0, 1]  # Get score for fake class (index 1)
    
    # Backward pass
    score.backward()
    
    # Remove hooks
    handle1.remove()
    handle2.remove()
    
    # Get the gradient and activation
    if len(gradients) == 0 or len(activations) == 0:
        # Fallback if GradCAM fails
        img_array = np.array(image.resize(IMAGE_SIZE))
        gradcam_filename = f"gradcam_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        return gradcam_path
    
    try:
        # For Swin Transformer, the output may need reshaping
        activation = activations[0]
        gradient = gradients[0]
        
        # Print shape for debugging
        print(f"Activation shape: {activation.shape}")
        print(f"Gradient shape: {gradient.shape}")
        
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
            gradcam_filename = f"gradcam_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
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
        
        # Calculate importance weights - adapt based on actual gradient shape
        if len(gradient.shape) == 4:
            weights = torch.mean(gradient[0], dim=(1, 2))
        else:
            # Alternative approach if the shape is unexpected
            weights = torch.mean(gradient.view(gradient.size(0), -1), dim=1)
        
        # Create weighted activation map - adapt based on actual activation shape
        if len(activation.shape) == 4:
            cam = torch.zeros(activation.shape[2:], dtype=torch.float32, device=device)
            for i, w in enumerate(weights):
                cam += w * activation[0, i]
        else:
            # Alternative approach for unexpected shapes
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
        gradcam_filename = f"gradcam_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
        cv2.imwrite(gradcam_path, superimposed_img)
        
        return gradcam_path
        
    except Exception as e:
        print(f"Error in GradCAM generation: {str(e)}")
        # Fallback - return original image with timestamp if GradCAM fails
        img_array = np.array(image.resize(IMAGE_SIZE))
        gradcam_filename = f"gradcam_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        return gradcam_path

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    original_filename = secure_filename(file.filename)

    # Open the image as a PIL Image
    image = Image.open(file.stream).convert("RGB")

    try:
        # Preprocess image
        img_tensor = preprocess_pil_image(image)
        
        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            prediction = probs[0][0].item()  # Changed to index 0 for Real probability
        
        # Determine Real/Fake (reversed from previous version)
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
        
        # Preprocess image
        img_tensor = preprocess_pil_image(image)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            prediction = probs[0][0].item()  # Changed to index 0 for Real probability
        
        # Determine Real/Fake (reversed from previous version)
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
        print(f"Error in /predict_url endpoint: {str(e)}")
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

def process_frame(frame, frame_count):
    """Process a single frame."""
    image = frame_to_image(frame)
    img_tensor = preprocess_pil_image(image)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        prediction = probs[0][0].item()  # Changed to index 0 for Real probability
    
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
        futures = {executor.submit(process_frame, frame, count): (frame, count) 
                  for frame, count in list(frame_queue.queue)}
        
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
        print(f"Error in /predict_video endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


"""
pip install flask tensorflow keras pillow
pip install opencv-python
pip install opencv-python-headless
pip install flask-cors
pip install  torch torchvision
"""