import pandas as pd
import os
import chardet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import cross-validation tools
from sklearn.model_selection import KFold, cross_val_score

# Construct the full file path
file_path = os.path.join(os.getcwd(), "SampleSalesData.csv")  


# Detect file encoding
try:
    with open(file_path, "rb") as f:
        raw_data = f.read(100000)  # Read first 100000 bytes for detection
        result = chardet.detect(raw_data)
        detected_encoding = result["encoding"]
except FileNotFoundError:
    print("Error: 'SampleSalesData.csv' not found. Checked path:", file_path)
    exit()

# Load the CSV file into a DataFrame
try:
    df = pd.read_csv(file_path, encoding=detected_encoding)
except Exception as e:
    print("Error loading file:", e)
    exit()

####################### Basic TEST exploration ##################
#print("\nFirst 5 rows of the DataFrame:")
#print(df.head())

#print("\nDataFrame information (data types, non-null counts):")
#print(df.info())

#print("\nSummary statistics for numerical columns:")
#print(df.describe())


######## Replace all NaN values with 0 #########
df.fillna(0, inplace=True)

######### Display dataset after filling NaN values #########
#print("\nAfter handling missing values:")
#print(df.info())  # Verify if NaN values are handled
#print(df.head())  # Check the updated data

######## Convert Dates to Proper Format ########

df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

######## Create new features that may help in prediction. For example, extract year and month ########

df['YEAR'] = df['ORDERDATE'].dt.year
df['MONTH'] = df['ORDERDATE'].dt.month


######## Drop Unnecessary Columns ########

df.drop(['CUSTOMERNAME', 'PHONE', 'ADDRESSLINE1', 'ADDRESSLINE2', 
         'CITY', 'STATE', 'POSTALCODE', 'COUNTRY', 'CONTACTLASTNAME', 
         'CONTACTFIRSTNAME', 'ORDERDATE'], axis=1, inplace=True) # Removed ORDERDATE

# Identify the problematic column:  Do this *before* dummies or train_test_split
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col]) #Try converting to numeric
        print(f"Column '{col}' can be converted to numeric.")
    except ValueError as e:
        print(f"Column '{col}' cannot be converted to numeric. Error: {e}")

# Handle the problematic columns.  Do this *before* get_dummies and train_test_split!
categorical_cols = ['STATUS', 'PRODUCTLINE', 'PRODUCTCODE', 'TERRITORY', 'DEALSIZE']  # List of ALL categorical columns
for col in categorical_cols:
    if col in df.columns:
        print(f"Handling {col} column...")
        df[col] = df[col].astype(str)
        #No need to get dummies these column. because we will do it later.
# One-Hot Encode Categorical Features (including STATUS, PRODUCTLINE, etc.)
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

######### Convert Categorical Columns to Numbers ########

#df = pd.get_dummies(df, columns=['STATUS', 'PRODUCTLINE', 'TERRITORY', 'DEALSIZE'], drop_first=True) #NO NEED FOR THAT


######### Define Features (X) and Target (y) ########

X = df.drop(['SALES'], axis=1)  # Features (everything except SALES)
y = df['SALES']  # Target (what we want to predict)

# Need to split data to 80% training and 20% testing here
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

########### training random forest Model ##########
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Train the model

y_pred = rf_model.predict(X_test)  # Predict sales

# Calculate error metrics
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 1. Scatter Plot of Actual vs. Predicted Values

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.title('Actual vs. Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Add a diagonal line for reference
plt.show()

# 2. Residual Plot

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.title('Residual Plot')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.axhline(y=0, color='k', linestyle='--', lw=2)  # Add a horizontal line at y=0
plt.show()

# ------------------ NEW PREDICTION -------------------------
# Get the correct column names from X_train
all_columns = X_train.columns

# Example values for the new data point:
quantity_ordered = 30
price_each = 50
order_line_number = 2
msrp = 100
year = 2024
month = 2
status = 'Shipped'          # Replace with a valid status from your dataset
productline = 'Motorcycles'   # Replace with a valid product line
productcode = 'S10_1678'       # Replace with a valid product code
territory = 'EMEA'          # Replace with a valid territory
dealsize = 'Small'          # Replace with a valid deal size

# Create a function to set one-hot encoded values correctly
def create_new_data(quantity_ordered, price_each, order_line_number, msrp, year, month,
                     status, productline, productcode, territory, dealsize, all_columns):
    new_data = pd.DataFrame(0, index=[0], columns=all_columns) #Index is necessary
    new_data['QUANTITYORDERED'] = quantity_ordered
    new_data['PRICEEACH'] = price_each
    new_data['ORDERLINENUMBER'] = order_line_number
    new_data['MSRP'] = msrp
    new_data['YEAR'] = year
    new_data['MONTH'] = month

    # One-hot encode categorical features
    new_data[f'STATUS_{status}'] = 1 if f'STATUS_{status}' in all_columns else 0
    new_data[f'PRODUCTLINE_{productline}'] = 1 if f'PRODUCTLINE_{productline}' in all_columns else 0
    new_data[f'PRODUCTCODE_{productcode}'] = 1 if f'PRODUCTCODE_{productcode}' in all_columns else 0
    new_data[f'TERRITORY_{territory}'] = 1 if f'TERRITORY_{territory}' in all_columns else 0
    new_data[f'DEALSIZE_{dealsize}'] = 1 if f'DEALSIZE_{dealsize}' in all_columns else 0
    return new_data
# Create new data with real value:
new_data = create_new_data(quantity_ordered, price_each, order_line_number, msrp, year, month,
                     status, productline, productcode, territory, dealsize, all_columns)
# Make prediction
predicted_sales = rf_model.predict(new_data)
print("Predicted Sales:", predicted_sales[0])