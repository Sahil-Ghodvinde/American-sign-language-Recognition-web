from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the trained CNN model
try:
    model = load_model('cnn_model.h5')
    print("CNN model loaded successfully")
except Exception as e:
    print("Warning: Could not load CNN model. Make sure cnn_model.h5 is in the correct location.", e)
    model = None

# Load the KNN model (optional)
try:
    with open('knn_model.pkl', 'rb') as f:
        knn = pickle.load(f)
    print("KNN model loaded successfully")
except Exception as e:
    print("Warning: Could not load KNN model. Make sure knn_model.pkl is in the correct location.", e)
    knn = None

# The class_names should match exactly what was used in training
# These should be the folder names in your dataset directory
class_names = ['bye', 'hello', 'no', 'perfect', 'thankyou', 'yes']

def preprocess_image(image_bytes):
    """ Convert image to the required format for the models """
    # Convert bytes data to OpenCV image
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28 to match the training dimensions
    processed = cv2.resize(gray, (28, 28)) / 255.0
    return processed, img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    if model is None and knn is None:
        return jsonify({'error': 'No models loaded'}), 500

    file = request.files['file']
    image_bytes = file.read()
    processed, original_img = preprocess_image(image_bytes)

    # Initialize results
    result = {'cnn': None, 'knn': None, 'consensus': None}
    
    # CNN prediction if model is available
    if model is not None:
        # Prepare image for CNN model: reshape to (1, 28, 28, 1)
        cnn_input = processed.reshape(1, 28, 28, 1)
        cnn_prediction = model.predict(cnn_input, verbose=0)
        cnn_label = int(np.argmax(cnn_prediction[0]))
        cnn_confidence = float(cnn_prediction[0][cnn_label])
        
        # Make sure the label is in range of class_names
        if 0 <= cnn_label < len(class_names):
            cnn_class = class_names[cnn_label]
        else:
            cnn_class = "Unknown"
            
        result['cnn'] = {
            'class': cnn_class,
            'confidence': cnn_confidence
        }

    # KNN prediction if model is available
    if knn is not None:
        # Prepare image for KNN model: flatten to (1, 784)
        knn_input = processed.reshape(1, -1)
        knn_label = knn.predict(knn_input)[0]
        knn_proba = knn.predict_proba(knn_input)[0]
        knn_confidence = float(knn_proba[knn_label])
        
        # Make sure the label is in range of class_names
        if 0 <= knn_label < len(class_names):
            knn_class = class_names[knn_label]
        else:
            knn_class = "Unknown"
            
        result['knn'] = {
            'class': knn_class,
            'confidence': knn_confidence
        }

    # Determine consensus if both models available
    if result['cnn'] and result['knn']:
        if result['cnn']['class'] == result['knn']['class']:
            consensus = result['cnn']['class']
            confidence = max(result['cnn']['confidence'], result['knn']['confidence'])
        else:
            # Use prediction with higher confidence
            if result['cnn']['confidence'] > result['knn']['confidence']:
                consensus = result['cnn']['class']
                confidence = result['cnn']['confidence']
            else:
                consensus = result['knn']['class']
                confidence = result['knn']['confidence']
        
        result['consensus'] = {
            'class': consensus,
            'confidence': confidence
        }
    # If only one model is available, use its result
    elif result['cnn']:
        result['consensus'] = result['cnn']
    elif result['knn']:
        result['consensus'] = result['knn']

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
