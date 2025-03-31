from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load your trained CNN model
# Note: You'll need to place your model file in this directory
try:
    model = load_model('cnn_model.h5')
    print("CNN model loaded successfully")
except:
    print("Warning: Could not load CNN model. Make sure cnn_model.h5 is in the correct location.")
    model = None

# Optionally load the KNN model (if you want to use it)
# try:
#     with open('knn_model.pkl', 'rb') as f:
#         knn = pickle.load(f)
#     print("KNN model loaded successfully")
# except:
#     print("Warning: Could not load KNN model. Make sure knn_model.pkl is in the correct location.")
#     knn = None

# Define label to letter mapping
label_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: 'Z'
}

def preprocess_image(image_bytes):
    # Convert bytes data to OpenCV image
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    # Convert to grayscale and resize to 28x28 as needed by your model
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.resize(gray, (28, 28)) / 255.0
    return processed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    file = request.files['file']
    image_bytes = file.read()
    processed = preprocess_image(image_bytes)

    # Prepare image for CNN model
    cnn_input = processed.reshape(1, 28, 28, 1)
    cnn_prediction = model.predict(cnn_input, verbose=0)
    cnn_label = int(np.argmax(cnn_prediction[0]))
    cnn_confidence = float(cnn_prediction[0][cnn_label])
    cnn_letter = label_to_letter.get(cnn_label, "Unknown")

    # Optionally, add KNN prediction here and choose the result with higher confidence

    # Return the prediction as JSON
    return jsonify({
        'letter': cnn_letter,
        'confidence': cnn_confidence
    })

if __name__ == '__main__':
    app.run(debug=True) 