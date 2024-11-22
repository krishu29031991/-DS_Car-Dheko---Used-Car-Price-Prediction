    import pandas as pd
    import ast

    # Load the Excel file
    #data_excel = pd.read_excel(r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\bangalore_cars.xlsx")
    #data_excel = pd.read_excel(r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\chennai_cars.xlsx")
    #data_excel = pd.read_excel(r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\delhi_cars.xlsx")
    #data_excel = pd.read_excel(r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\hyderabad_cars.xlsx")
    #data_excel = pd.read_excel(r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\jaipur_cars.xlsx")
    data_excel = pd.read_excel(r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\kolkata_cars.xlsx")


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


    #removing string 
    df["Acceleration"] = df["Acceleration"].str.extract(r'([\d]+\.[\d]+|\d+)')
    df["Alloy Wheel Size"]=df["Alloy Wheel Size"].str.extract(r'([\d]+)')
    df["Cargo Volumn_litres"]=df["Cargo Volumn"].str.extract(r'([\d]+)')
    df["Engine_cc"]=df["Engine"].str.extract(r'([\d]+)')
    df["price_in_lakh"]=df["price"].str.extract(r'([\d]+\.[\d]+|\d+)')
    df["priceActual_in_lakh"]=df["priceActual"].str.extract(r'([\d]+\.[d]+|\d+)')
    df["Registration Year"]=df["Registration Year"].str.extract(r'([\d]+)')
    df["Front Tread_in_mm"]=df["Front Tread"].str.extract(r'([\d]+)')
    df["Mileage_in_kmpl"]=df["Mileage"].str.extract(r'([\d.]+)').astype(float)
    df['Kms Driven_kms'] = df['Kms Driven'].str.extract(r'([\d,]+)')[0].str.replace(',', '').astype(float).map('{:.2f}'.format)
    df["Torque"]=df["Torque"].str.extract(r'([\d]+)')

    #Assigning value to string
    df["Insurance Validity"]=df["Insurance Validity"].map({"Zero Dep":0,"Third Party insurance":3,"Third Party":3,"Comprehensive":4,"1":1,"2":2})
    df['transmission_code'] = df['transmission'].map({'Manual': 1, 'Automatic': 2})
    df["Fuel Type_no"]=df["Fuel Type"].map({"Petrol":1,"Diesel":2,"Electric":3,"LPG":4,"CNG":5})
    df["Steering Type"]=df["Steering Type"].map({"Manual":1, "Power":2,"Electric":3,"Electrical":3,"EPAS":3,"Electronic":4})


    #Removing Unnecessary columns
    df.drop(columns=["Cargo Volumn","car_links","bt","Engine Displacement","Gear Box","Color","Max Power","Max Torque",
                     "features_top","model","oem","trendingText","variantName","Engine","Transmission","ft",
                     "price","priceActual","Front Tread","Mileage","Kms Driven","BoreX Stroke",
                     "Comfort & Convenience","Compression Ratio","Drive Type","Engine Type","transmission",
                     "Entertainment & Communication","Exterior","Front Brake Type","Wheel Base","Wheel Size","Width",
                     "Fuel Suppy System","Gross Weight","Ground Clearance Unladen","Turbo Charger","Turning Radius","Tyre Type","Value Configuration",
                     "Height","Interior","Kerb Weight","Length","RTO","Rear Brake Type","Ownership","Super Charger","Top Speed",
                     "Rear Tread","Safety","transmission","priceSaving","priceFixedText","owner","it","Fuel Type"],inplace=True)


    #Convert Datatypes
    df["Acceleration"]=df["Acceleration"].astype(float)
    df["Alloy Wheel Size"]=df["Alloy Wheel Size"].fillna(0).astype(int)
    df["Displacement"]=df["Displacement"].fillna(0).astype(int)
    df["Insurance Validity"]=df["Insurance Validity"].fillna(0).astype(int)
    df["No Door Numbers"]=df["No Door Numbers"].fillna(0).astype(int)
    df["No of Cylinder"]=df["No of Cylinder"].fillna(0).astype(int)
    df["Registration Year"] = df["Registration Year"].fillna(0).astype(int)

    df["Seats"] = df["Seats"].replace('null', pd.NA)  # Replace 'null' with NaN
    df["Seats"] = pd.to_numeric(df["Seats"], errors='coerce').fillna(0).astype(int)
    #df["Seats"]=df["Seats"].fillna(0).astype(int)
    df["Steering Type"]=df["Steering Type"].fillna(0).astype(int)
    df["Values per Cylinder"]=df["Values per Cylinder"].fillna(0).astype(int)
    df["km"]=df["km"].str.replace(',',"").fillna(0).astype(int)
    df["ownerNo"]=df["ownerNo"].fillna(0).astype(int)
    df["Engine_cc"]=df["Engine_cc"].fillna(0).astype(int)
    df["price_in_lakh"]=df["price_in_lakh"].fillna(0).astype(float)
    df["Mileage_in_kmpl"]=df["Mileage_in_kmpl"].fillna(0).astype(float)
    df["Kms Driven_kms"]=df["Kms Driven_kms"].astype(float)
    df["transmission_code"]=df["transmission_code"].fillna(0).astype(int)
    df["Fuel Type_no"]=df["Fuel Type_no"].fillna(0).astype(int)


    # Save the modified DataFrame to a new Excel file
    #output_path = r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\bangalore_cars_testing.xlsx"
    #output_path = r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\chennai_cars_testing.xlsx"
    #output_path = r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\delhi_cars_testing.xlsx"
    #output_path = r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\hyderabad_cars_testing.xlsx"
    #output_path = r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\jaipur_cars_testing.xlsx"
    output_path = r"C:\Users\10732370\OneDrive - LTIMindtree\Documents\guvi\Project 3 Cardheko\kolkata_cars_testing.xlsx"

    df.to_excel(output_path, index=False)

    print(f"The modified DataFrame has been saved to {output_path}.")
