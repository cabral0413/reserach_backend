from flask import Flask, request, jsonify
from PIL import Image, ImageOps
from flask_cors import CORS
import io
import os
import onnxruntime as ort
import numpy as np

app = Flask(__name__)
CORS(app)

# Load ONNX models during startup
onnx_detection_model_path = 'models/best.onnx'  # ONNX model for object detection
onnx_classification_model_path = 'models/bestc.onnx'  # ONNX model for classification

detection_session = ort.InferenceSession(onnx_detection_model_path)
classification_session = ort.InferenceSession(onnx_classification_model_path)

CONFIDENCE_THRESHOLD = 0.5  # Set a confidence threshold for valid detections

def preprocess_image(image):
    """ Preprocess the image for inference. Convert to numpy array and normalize. """
    image = image.resize((640, 640))  # Resize to 640x640 (YOLO size)
    img_np = np.array(image).astype(np.float32)
    img_np = np.transpose(img_np, (2, 0, 1)) / 255.0  # Normalize and convert to CHW format
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

@app.route('/classify', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert("RGB")  # Ensure it's in RGB format

        # Preprocess the image for object detection
        img_np = preprocess_image(img)

        # Run ONNX detection model
        detection_inputs = {detection_session.get_inputs()[0].name: img_np}
        detection_results = detection_session.run(None, detection_inputs)

        # Process detection results (assuming results are in a specific format)
        boxes, scores, labels = detection_results[0], detection_results[1], detection_results[2]
        
        # Filter out detections below confidence threshold
        valid_detections = [(box, score, label) for box, score, label in zip(boxes, scores, labels)
                            if score > CONFIDENCE_THRESHOLD]

        if len(valid_detections) == 0:
            return jsonify({'message': 'No objects detected in the image.'}), 200

        # Select the best detection
        best_box, best_score, best_label = max(valid_detections, key=lambda x: x[1])

        # Crop and preprocess the image for classification
        box_coords = best_box.tolist()
        cropped_img = img.crop(box_coords)  # Crop using bounding box coordinates
        padded_img = ImageOps.pad(cropped_img, (224, 224), method=Image.Resampling.LANCZOS)  # Resize to 224x224
        
        # Preprocess the cropped image for classification
        padded_img_np = preprocess_image(padded_img)

        # Run ONNX classification model
        classification_inputs = {classification_session.get_inputs()[0].name: padded_img_np}
        classification_results = classification_session.run(None, classification_inputs)

        # Assuming classification result is a softmax vector
        probabilities = classification_results[0]
        top_class_idx = np.argmax(probabilities)
        top_confidence = probabilities[0][top_class_idx]
        
        # If the classification confidence is below threshold, reject the result
        if top_confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'message': 'No valid snake detected. Please upload a clearer image.',
                'class': 'Unknown',
                'probability': "{:.2%}".format(top_confidence)
            }), 200

        # Get the class name based on top_class_idx (assuming a fixed class list)
        class_names = ['Common Indian Krait', 'Python', 'Hump Nosed Viper', 'Green Vine Snake', 'Russells Viper', 'Indian Cobra']
        top_class = class_names[top_class_idx]

        # Determine venom status
        venom_status = get_venom_status(top_class)

        # Format the probability as a percentage with two decimal points
        formatted_prob = "{:.2%}".format(top_confidence)

        # Return the prediction
        return jsonify({
            'predictions': [{
                'class': top_class,
                'probability': formatted_prob,
                'venom_status': venom_status
            }]
        }), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'An error occurred during processing.', 'details': str(e)}), 500

def get_venom_status(class_name):
    venom_status_map = {
        'Common Indian Krait': 'Venomous',
        'Python': 'Non-venomous',
        'Hump Nosed Viper': 'Venomous',
        'Green Vine Snake': 'Non-venomous',
        'Russells Viper': 'Venomous',
        'Indian Cobra': 'Venomous'
    }
    return venom_status_map.get(class_name, 'Unknown')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
