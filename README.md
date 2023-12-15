# Color Recognition API with Flask, Google Cloud and TensorFlow

This repository contains a simple Flask web application that serves as an API for color recognition. The application uses pre-trained TensorFlow models to predict the color of objects in uploaded images.

## Getting Started

### Prerequisites
- Flask
- gunicorn
- tensorflow
- numpy
- pillow
- flask_cors

### Installation
1. Clone the repository: `https://github.com/ibnuarz/ColorRecognition_API_TensorFlow.git`
2. Install dependencies: `pip install -r requirements.txt`

### Usage
1. Run the Flask application: `python precolor.py`
2. Upload an image file using the provided HTML form or make POST requests to the `/` endpoint.
3. Receive JSON responses with color predictions.
4. If you run on local you can use local server example `http://127.0.0.1:5000/` but in this case i use endpoint from google cloud

### Important
1. You can use my model or your model to predict (CNN) , here my model `https://drive.google.com/file/d/1-0BJrIl491gnEulSv4-0j3TSj6ZuruK4/view?usp=sharing`

## License
This project is licensed under the MIT License
