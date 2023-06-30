import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loading dataset to a Pandas DataFrame
wine_dataset = pd.read_csv('winequality-red.csv')

# Separate the data and label
X = wine_dataset.drop('quality', axis=1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Function to predict wine quality
def predict_quality(input_data):
    # Reshape the input data
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    # Make prediction
    prediction = model.predict(input_data_reshaped)
    # Return the prediction
    return prediction[0]

# Streamlit web application
def main():
    # Page title
    st.title("Wine Quality Prediction")

    # Input section
    st.header("Enter Wine Features:")
    input_data = []
    for feature in X.columns:
        value = st.number_input(f"{feature}:")
        input_data.append(value)

    # Predict button
    if st.button("Predict"):
        # Make prediction
        prediction = predict_quality(input_data)
        # Display prediction result
        if prediction == 1:
            st.success("Good Quality Wine")
        else:
            st.error("Bad Quality Wine")

# Run the web application
if __name__ == '__main__':
    main()
