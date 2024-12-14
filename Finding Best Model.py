import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from the Excel file
data = pd.read_excel('all_cars_testing.xlsx')

# Define the features (X) and the target variable (y)
X = data.drop(columns=['price_in_lakh'])
y = data['price_in_lakh']

# Handle missing values by filling them with the mean of each column
X = X.fillna(X.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Support Vector Machine': SVR(),
    'Neural Network': MLPRegressor(random_state=42, max_iter=1000)
}

# Initialize variables to store the best model and its corresponding metrics
best_model_name = None
best_mse = float('inf')
best_rmse = float('inf')
best_r2 = float('-inf')

# Evaluate each model
for model_name, model in models.items():
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the testing data
    y_pred = model.predict(X_test)
    
    # Calculate the mean squared error, root mean squared error, and R-squared
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    # Print the performance metrics for the current model
    print(f"{model_name}:")
    print(f"  Mean Squared Error (MSE): {mse}")
    print(f"  Root Mean Squared Error (RMSE): {rmse}")
    print(f"  R-squared (R²): {r2}")
    
    # Update the best model if the current one is better
    if mse < best_mse:
        best_model_name = model_name
        best_mse = mse
        best_rmse = rmse
        best_r2 = r2

# Print the best model and its corresponding metrics
print(f"\nBest Model: {best_model_name}")
print(f"Mean Squared Error (MSE): {best_mse}")
print(f"Root Mean Squared Error (RMSE): {best_rmse}")
print(f"R-squared (R²): {best_r2}")

