import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from the Excel file
data = pd.read_excel(r'C:\Users\avr81\Documents\Guvi\Cardheko project 3\Extracted data\combined_excel.xlsx')

# Define the features (X) and the target variable (y)
X = data.drop(columns=['price_in_lakh'])
y = data['price_in_lakh']

# Handle missing values by filling them with the mean of each column
X = X.fillna(X.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=93)

# Create a Gradient Boosting Regressor model
model = GradientBoostingRegressor()

# Train the model on the training data
model.fit(X_train, y_train)

# Streamlit app
st.title('Car Price Prediction')

# List of columns that should be integers
integer_columns = ['Alloy Wheel Size', 'Displacement', 'Insurance Validity', 'No Door Numbers','No of Cylinder',
                   'Registration Year','Seating Capacity','Seats','Steering Type','Torque','Values per Cylinder','Year of Manufacture',
                   'centralVariantId','km','modelYear','ownerNo','Cargo Volumn_litres','Engine_cc','Front Tread_in_mm',
                   'Mileage_in_kmpl','Kms Driven_kms','Fuel Type_no']

# Dynamically generate input fields for each column in X
inputs = {}
for column in X.columns:
    if column in integer_columns:
        # Ensure integer input for specific columns, round to nearest integer
        inputs[column] = st.number_input(f'Enter {column}', value=int(round(X[column].mean())), step=1)
        inputs[column] = int(inputs[column])  # Explicitly cast to int after input
    else:
        # Float input for other columns
        inputs[column] = st.number_input(f'Enter {column}', value=float(X[column].mean()), step=0.01)

# Convert inputs to a DataFrame for prediction
input_df = pd.DataFrame([inputs])

# Explicitly cast integer columns to int
for column in integer_columns:
    if column in input_df.columns:
        input_df[column] = input_df[column].astype(int)

# Predict button
if st.button('Predict Price'):
    try:
        # Make prediction
        prediction = model.predict(input_df)[0]

        # Display prediction
        st.success(f'Predicted Car Price: ₹{prediction:.2f} lakh')
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Display model performance metrics
st.write('### Model Performance Metrics')
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
st.write(f'Mean Squared Error (MSE): {mse}')
st.write(f'Root Mean Squared Error (RMSE): {rmse}')
st.write(f'R-squared (R²): {r2}')

# Debugging: Display the expected and actual input DataFrame
st.write('### Debugging Information')
st.write("Input DataFrame:", input_df)
st.write("Expected Columns:", X.columns)
