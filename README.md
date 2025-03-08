# Deep Fake Detection for Images and Videos


## Features
- Detect deep fakes in images and videos.
- RESTful API to interact with the detection model.
- CORS enabled for cross-origin resource sharing.

## Prerequisites
- Python 3.12

## Installation

### 1. Clone the Repository

### 2. Create a Virtual Environment
```bash
python -m venv myenv
```

### 3. Activate the Virtual Environment
```bash
myenv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install flask tensorflow keras pillow
pip install opencv-python
pip install opencv-python-headless
pip install flask-cors
pip install torch torchvision
```

### 5. Run the Application
```bash
python app.py
```

## API Documentation
You can find the API documentation [here](https://documenter.getpostman.com/view/24694319/2sAYX9o1Vj).

## Models
This project utilizes the following deep learning models for detection:

- **Xception Model** → [Download Here](https://www.kaggle.com/code/abdelrahmantarekm/xception-final-df/output)
- **Swin Model** → [Download Here](https://www.kaggle.com/code/abdelrahmanhesham101/swin-transformer/output?scriptVersionId=224698098)

