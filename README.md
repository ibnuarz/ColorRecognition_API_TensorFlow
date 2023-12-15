# Color Recognition API with Flask and TensorFlow

This repository contains a simple Flask web application that serves as an API for color recognition. It utilizes a pre-trained TensorFlow model to predict the color of objects in uploaded images.

## Getting Started

### Prerequisites
- Flask
- gunicorn
- tensorflow
- numpy
- pillow
- flask_cors

### Installation
1. Clone the repository: `-`
2. Install dependencies: `pip install -r requirements.txt`

### Usage
1. Run the Flask application: `python precolor.py`
2. Upload an image file using the provided HTML form or make POST requests to the `/` endpoint.
3. Receive JSON responses with color predictions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
