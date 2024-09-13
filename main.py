import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
file_path = '/content/Housing.csv'
data = pd.read_csv(file_path)


# Keep only the specified columns
columns_to_keep = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
filtered_data = data[columns_to_keep]

# Display the first few rows of the filtered dataset
print(filtered_data.head())

# Features and Target
X = filtered_data.drop('price', axis=1)  # Features (drop the target column)
y = filtered_data['price']               # Target (House Prices)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model to a file
joblib.dump(model, 'model.pkl')
