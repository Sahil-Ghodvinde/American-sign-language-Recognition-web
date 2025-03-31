# ASL Recognition Web Application

This web application allows users to recognize American Sign Language (ASL) gestures using a pre-trained neural network. Users can either upload an image or use their webcam to capture ASL gestures, and the application will predict the corresponding letter.

## Prerequisites

- Python 3.7+
- Flask
- TensorFlow
- OpenCV
- NumPy

## Setup Instructions

1. Make sure all required packages are installed:
   ```
   pip install flask tensorflow opencv-python numpy
   ```

2. Place your pre-trained models in the project root directory:
   - `cnn_model.h5`: The trained CNN model
   - `knn_model.pkl` (optional): If you're using a KNN model

3. Run the Flask application:
   ```
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

- Click "Use Webcam" to access your webcam, then "Capture Image" to take a photo of your ASL gesture.
- Alternatively, click "Upload Image" to select a saved image of an ASL gesture.
- The application will process the image and display the predicted letter along with the confidence score.

## Model Information

The application uses a Convolutional Neural Network (CNN) trained on the ASL alphabet dataset. The model recognizes 24 letters (J and Z are excluded as they involve motion).

## Deployment

To deploy the application to a production environment:

1. Configure the appropriate port and host settings in app.py
2. Set debug=False for production
3. Consider using a WSGI server like Gunicorn or uWSGI
4. Follow the hosting platform's documentation for deploying Flask applications 