import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from the Excel file
data = pd.read_excel('all_cars_testing.xlsx')

# Define the features (X) and the target variable (y)
X = data.drop(columns=['price_in_lakh'])
y = data['price_in_lakh']

# Handle missing values by filling them with the mean of each column
X = X.fillna(X.mean())

# Initialize variables to store the best random_state and its corresponding metrics
best_random_state = None
best_mse = float('inf')
best_rmse = float('inf')
best_r2 = float('-inf')

# Tune the random_state parameter
for random_state in range(1, 101):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the mean squared error of the predictions
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Update the best random_state if current one is better
    if mse < best_mse:
        best_random_state = random_state
        best_mse = mse
        best_rmse = rmse
        best_r2 = r2

# Print the best random_state and its corresponding metrics
print(f"Best Random State: {best_random_state}")
print(f"Mean Squared Error (MSE): {best_mse}")
print(f"Root Mean Squared Error (RMSE): {best_rmse}")
print(f"R-squared (R²): {best_r2}")
