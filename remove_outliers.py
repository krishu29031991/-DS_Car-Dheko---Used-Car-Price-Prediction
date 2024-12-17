import pandas as pd

# Load the dataset
df = pd.read_excel('testing_check9.xlsx')

# Identify numerical columns
numerical_cols = df.select_dtypes(exclude=['object']).columns.tolist()

# Remove outliers using the IQR method for numerical columns only
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df[~((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Save the cleaned dataset to a new Excel file
df_no_outliers.to_excel('cleaned_testing_check9.xlsx', index=False)

print("Outliers removed and cleaned dataset saved to 'cleaned_testing_check9.xlsx'.")
