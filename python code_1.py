import pandas as pd
import ast

# Load the Excel file
data_excel = pd.read_excel(r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\bangalore_cars.xlsx")

# Function to parse new_car_detail
def parse_new_car_detail(row):
    try:
        dict_data = ast.literal_eval(row['new_car_detail'])
        for key, value in dict_data.items():
            row[key] = value
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing JSON for row: {row['new_car_detail']} - {e}")
    return row

# Function to parse new_car_overview
def parse_new_car_overview(row):
    try:
        dict_data = ast.literal_eval(row['new_car_overview'])
        if 'top' in dict_data:
            for item in dict_data['top']:
                key = item['key']
                value = item['value']
                row[key] = value
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing JSON for row: {row['new_car_overview']} - {e}")
    return row

# Function to parse new_car_feature
def parse_new_car_feature(row):
    try:
        dict_data = ast.literal_eval(row['new_car_feature'])
        features_top = [item['value'] for item in dict_data.get('top', [])]
        row['features_top'] = ', '.join(features_top)
        for section in dict_data.get('data', []):
            heading = section.get('heading', '')
            features_data = [item['value'] for item in section.get('list', [])]
            row[heading] = ', '.join(features_data)
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing JSON for row: {row['new_car_feature']} - {e}")
    return row

# Function to parse new_car_specs
def parse_new_car_specs(row):
    try:
        dict_data = ast.literal_eval(row['new_car_specs'])
        for item in dict_data.get('top', []):
            key = item['key']
            value = item['value']
            row[key] = value
        for section in dict_data.get('data', []):
            for item in section.get('list', []):
                key = item['key']
                value = item['value']
                row[key] = value
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing JSON for row: {row['new_car_specs']} - {e}")
    return row

# Apply each function to the DataFrame
df = data_excel.copy()
df = df.apply(parse_new_car_detail, axis=1)
df = df.apply(parse_new_car_overview, axis=1)
df = df.apply(parse_new_car_feature, axis=1)
df = df.apply(parse_new_car_specs, axis=1)

# Remove the specified columns
columns_to_remove = ['new_car_detail', 'new_car_overview', 'new_car_feature', 'new_car_specs']
df.drop(columns=columns_to_remove, inplace=True)

# Save the modified DataFrame to a new Excel file
output_path = r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\bangalore_cars_testing.xlsx"
df.to_excel(output_path, index=False)

print(f"The modified DataFrame has been saved to {output_path}.")
