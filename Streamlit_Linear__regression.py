# prompt: centralVariantId,No of Cylinder,Engine_cc,maxbhppower,maxrpmpower,transmission_code,Fuel Type_no.... remove this column from streamlit UI

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Load the dataset
try:
    df = pd.read_excel('testing_check6.xlsx')
except FileNotFoundError:
    st.error("Error: 'testing_check6.xlsx' not found. Please upload the file to your Streamlit environment or adjust the file path.")
    exit()

# Check for missing values and handle them
if df.isnull().sum().sum() > 0:
    st.warning("Dataset contains missing values. Filling them with forward fill.")
    df.fillna(method='ffill', inplace=True)

# Feature Engineering and Model Training
X = df.drop('price_in_lakh', axis=1)
y = df['price_in_lakh']

# Remove specified columns
columns_to_remove = ['centralVariantId', 'No of Cylinder', 'Engine_cc', 'maxbhppower', 'maxrpmpower', 'transmission_code', 'Fuel Type_no']
for col in columns_to_remove:
    if col in X.columns:
        X = X.drop(col, axis=1)
    else:
        print(f"Warning: Column '{col}' not found in the dataset.")


X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Streamlit UI
st.title("Used Car Price Prediction")

# Input features
st.sidebar.header("Input Features")

# Create input fields for each feature
input_features = {}
for col in X.columns:
    if 'Insurance Validity' in col:
        input_features[col] = st.sidebar.number_input(col, min_value=0)
    elif 'No Door Numbers' in col:
        input_features[col] = st.sidebar.number_input(col, min_value=0)
    elif col == 'Seats':
        input_features[col] = st.sidebar.number_input(col, min_value=1)
    elif 'modelYear' in col:
        input_features[col] = st.sidebar.number_input(col, min_value=1990)
    else:
        input_features[col] = st.sidebar.selectbox(col, options=df[col].unique())

# Create a DataFrame from the input features
input_df = pd.DataFrame([input_features])

# Ensure the input DataFrame has the same columns as the training data
input_df = input_df.reindex(columns=X.columns, fill_value=0)


# Make Prediction
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)
    st.write(f"Predicted Price: {prediction[0]:.2f} Lakh")

st.write("Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"R² Score: {r2:.4f}")
