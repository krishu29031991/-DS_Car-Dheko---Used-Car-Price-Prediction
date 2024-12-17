import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

data = load_data(r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\cleaned_testing_check9.xlsx")

# Define the target and features
target = 'price_in_lakh'
features = data.columns.drop([target])

# One-hot encode categorical features
data_encoded = pd.get_dummies(data, columns=data.select_dtypes(include=['object']).columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_encoded.drop(columns=[target]), data_encoded[target], test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a Gradient Boosting Regressor model
model = GradientBoostingRegressor()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Streamlit app
st.title('Car Price Prediction')

# User input for prediction with drop-downs populated with unique values from the existing data
user_input = {}
filtered_data = data.copy()

# Filtered columns for car model details
filtered_columns = ['oem', 'model', 'bt']

# Columns to exclude from Streamlit UI
exclude_columns = ['Engine_cc', 'ownerNo', 'transmission_code', 'Fuel Type_no', 'centralVariantId', 'No of Cylinder', 'car_links', 'features_top', 'trendingText', 'maxbhppower', 'maxrpmpower']

# Define ranges for specific columns
ranges = {
    'Insurance Validity': (0, 10),
    'No Door Numbers': (0, 20),
    'Seats': (0, 30),
    'modelYear': (1980, 2024),
    'owner': (0, 10),
    'Mileage_in_kmpl': (0, 100),  # Assuming a reasonable range for mileage
    'Kms Driven_kms': (0, 1000000)  # Assuming a reasonable range for kilometers driven
}

for feature in features:
    if feature in filtered_columns:
        unique_values = sorted(filtered_data[feature].astype(str).unique())
        user_input[feature] = st.selectbox(f'Select {feature}', unique_values)
        filtered_data = filtered_data[filtered_data[feature].astype(str) == user_input[feature]]
    elif feature in ranges:
        user_input[feature] = st.slider(f'Select {feature}', min_value=ranges[feature][0], max_value=ranges[feature][1])
    elif feature not in exclude_columns:
        unique_values = sorted(data[feature].astype(str).unique())
        user_input[feature] = st.selectbox(f'Select {feature}', unique_values)

if st.button('Predict'):
    user_input_df = pd.DataFrame([user_input])
    user_input_encoded = pd.get_dummies(user_input_df)
    user_input_encoded = user_input_encoded.reindex(columns=X_train.columns, fill_value=0)
    user_input_scaled = scaler.transform(user_input_encoded)
    prediction = model.predict(user_input_scaled)
    st.write('Predicted Price in Lakh:', prediction[0])

# Display model performance metrics
st.write('### Model Performance Metrics')
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
st.write(f'Mean Squared Error (MSE): {mse}')
st.write(f'Root Mean Squared Error (RMSE): {rmse}')
st.write(f'R-squared (R²): {r2}')
