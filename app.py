from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
        if not all(feature in data for feature in required_features):
            return jsonify({'status': 'error', 'message': 'Missing required features'}), 400

        features = np.array([[
            data['area'],
            data['bedrooms'],
            data['bathrooms'],
            data['stories'],
            data['parking']
        ]])

        prediction = model.predict(features)[0]

        return jsonify({'status': 'success', 'predicted_price': prediction})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)