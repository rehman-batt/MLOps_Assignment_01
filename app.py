from flask import Flask, request, jsonify, render_template
import numpy as np
from main import MultilinearRegression
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
 
        # Ensure the correct features are provided
        required_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
        if not all(feature in data for feature in required_features):
            return jsonify({'status': 'error', 'message': 'Missing required features'}), 400

        # Extract features and convert them to numpy array
        features = np.array([[
            data['area'],
            data['bedrooms'],
            data['bathrooms'],
            data['stories'],
            data['parking']
        ]])

        # Make prediction
        prediction = model.predict(features)[0]

        return jsonify({'status': 'success', 'predicted_price': prediction})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run()