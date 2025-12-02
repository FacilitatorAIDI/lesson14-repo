import os
import pickle
from flask import Flask, request, jsonify
import numpy as np

# Set the model path relative to the app's location
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')

# Load the model
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialise the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Scikit-Learn Iris Model API is running."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True)

        # The model expects a 2D array, e.g., [[5.1, 3.5, 1.4, 0.2]]
        features = np.array(data)

        if features.ndim!= 2 or features.shape!= 4:
            return jsonify({'error': 'Input data must be a 2D array with 4 features.'}), 400

        # Make prediction
        prediction = model.predict(features)

        # Return prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # This part is for local debugging, not for production
    app.run(debug=True, host='0.0.0.0', port=5000) 

