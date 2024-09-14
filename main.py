import numpy as np
import pandas as pd
import pickle

class MultilinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        # Calculate the coefficients using the Normal Equation
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.coefficients)

    def score(self, X, y):
        y_pred = self.predict(X)
        # Calculate R^2 score
        total_variance = np.sum((y - np.mean(y)) ** 2)
        explained_variance = np.sum((y_pred - y) ** 2)
        return 1 - (explained_variance / total_variance)

def train_model():
    # Load the data
    data = pd.read_csv(r"D:\Fast NU\7th semester (pro max plus) shit\MLOps\Assignment_01\Data\Housing.csv")

    # Select relevant features for the model
    X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']].values

    # Select the target variable (label)
    y = data['price'].values

    # Instantiate and train the multilinear regression model
    model = MultilinearRegression()
    model.fit(X, y)

    # Make predictions on the same data (you can also use different test data)
    predictions = model.predict(X)

    # Print predictions and model performance
    print("Predictions:", predictions[:5])  # Printing first 5 predictions for brevity
    print("R^2 Score:", model.score(X, y))

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    train_model()