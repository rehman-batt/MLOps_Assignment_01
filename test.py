import numpy as np
import pytest
import joblib
 

def test_prediction_output_type():
    sample_input = np.array([[1500, 3, 2, 2, 1]]) 
    prediction = joblib.load('model.pkl').predict(sample_input)
    
    assert isinstance(prediction, np.ndarray)
    assert isinstance(prediction[0], (int, float))

def test_zero_input():
    sample_input = np.array([[0, 0, 0, 0, 0]])  # Zero values input
    prediction = joblib.load('model.pkl').predict(sample_input)
    
    assert np.isfinite(prediction).all()

def test_invalid_input_shape():
    sample_input = np.array([[1500, 3, 2]])  
    with pytest.raises(ValueError):
        joblib.load('model.pkl').predict(sample_input)


