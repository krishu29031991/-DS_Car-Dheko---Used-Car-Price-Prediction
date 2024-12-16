# prompt: check the data.. testing_check6.xlsx and predict the price_in_lakh

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
try:
    df = pd.read_excel('testing_check6.xlsx')
except FileNotFoundError:
    print("Error: 'testing_check6.xlsx' not found. Please upload the file to your Colab environment.")
    exit()


# Assuming 'price_in_lakh' is the target variable and other columns are features
# Identify features (X) and target (y)
X = df.drop('price_in_lakh', axis=1)  # Features
y = df['price_in_lakh']  # Target variable

# Handle missing values (if any) - simple imputation for demonstration
#X.fillna(X.mean(), inplace=True) # Replace with more sophisticated methods if needed

# Convert non-numeric columns to numerical representations (if any) using one-hot encoding
X = pd.get_dummies(X, drop_first=True) # Avoid multicollinearity


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict on the entire dataset to get predictions for all rows
all_predictions = model.predict(X)
df['predicted_price_in_lakh'] = all_predictions

# Print or save the results (showing predicted prices)
print(df[['price_in_lakh', 'predicted_price_in_lakh']])

# Optionally, save the results to a new Excel file
# df.to_excel('predictions.xlsx', index=False)
