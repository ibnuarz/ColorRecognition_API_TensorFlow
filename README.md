# Color Recognition API with Flask, Google Cloud and TensorFlow

Repositori ini berisi aplikasi web Flask sederhana yang berfungsi sebagai API untuk pengenalan warna. Aplikasi ini menggunakan model TensorFlow yang telah dilatih sebelumnya untuk memprediksi warna objek dalam gambar yang diunggah. 

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

## License
This project is licensed under the MIT License
