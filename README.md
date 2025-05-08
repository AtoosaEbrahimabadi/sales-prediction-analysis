# sales-prediction-analysis
Sales analysis and prediction using Python, Pandas, and Scikit-learn


# Sales Prediction Analysis

## Project Overview

This project focuses on analyzing historical sales data to understand key factors influencing sales and to build a machine learning model capable of predicting future sales. The analysis involves data loading, cleaning, feature engineering, model training (using Random Forest Regressor), evaluation, and a mechanism for predicting sales based on new input data.

## Goals

The primary goals of this project are:
1.  **Load and Preprocess Data:** Securely load the `SampleSalesData.csv` dataset, handling potential encoding issues and missing values.
2.  **Feature Engineering:** Extract meaningful features from existing data (e.g., year and month from order dates) and transform categorical features into a numerical format suitable for machine learning (via one-hot encoding).
3.  **Model Training:** Develop a predictive model using a Random Forest Regressor to forecast sales.
4.  **Model Evaluation:** Assess the performance of the trained model using standard regression metrics (MAE, MSE, R² score) and visualizations (Actual vs. Predicted, Residual Plot).
5.  **New Sales Prediction:** Provide a clear method to predict sales for new, unseen data points using the trained model.

## Dataset

The project utilizes `SampleSalesData.csv`, which contains historical sales records. Key features include:
*   `ORDERNUMBER`
*   `QUANTITYORDERED`
*   `PRICEEACH`
*   `ORDERLINENUMBER`
*   `SALES` (Target Variable)
*   `ORDERDATE`
*   `STATUS`
*   `QTR_ID`
*   `MONTH_ID`
*   `YEAR_ID`
*   `PRODUCTLINE`
*   `MSRP`
*   `PRODUCTCODE`
*   `CUSTOMERNAME` (dropped)
*   `PHONE` (dropped)
*   `ADDRESSLINE1`, `ADDRESSLINE2` (dropped)
*   `CITY`, `STATE`, `POSTALCODE`, `COUNTRY` (dropped)
*   `TERRITORY`
*   `CONTACTLASTNAME`, `CONTACTFIRSTNAME` (dropped)
*   `DEALSIZE`

## Methodology

The project follows these main steps:
1.  **Data Loading:** The `SampleSalesData.csv` is loaded, with automatic detection of file encoding.
2.  **Data Cleaning:**
    *   Missing values (NaN) are filled with 0.
    *   `ORDERDATE` is converted to a datetime object.
3.  **Feature Engineering:**
    *   `YEAR` and `MONTH` are extracted from `ORDERDATE`.
    *   Irrelevant columns (customer details, addresses, original `ORDERDATE`) are dropped.
4.  **Categorical Feature Encoding:**
    *   Categorical columns (`STATUS`, `PRODUCTLINE`, `PRODUCTCODE`, `TERRITORY`, `DEALSIZE`) are identified and converted to string type.
    *   One-hot encoding is applied to these categorical features using `pd.get_dummies()`.
5.  **Model Training & Evaluation:**
    *   The data is split into training (80%) and testing (20%) sets.
    *   A Random Forest Regressor model is trained on the training data.
    *   The model's performance is evaluated on the test data using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score.
    *   Visualizations (Actual vs. Predicted scatter plot, Residual plot) are generated to further assess model performance.
6.  **Prediction on New Data:**
    *   A function `create_new_data` is provided to format new input data (including one-hot encoding) to match the features expected by the trained model.
    *   An example demonstrates how to make a sales prediction for a new data point.

## Technologies Used

*   **Python 3.x**
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Matplotlib & Seaborn:** For data visualization.
*   **Scikit-learn:** For machine learning tasks (model selection, Random Forest, metrics).
*   **Chardet:** For character encoding detection.

## Setup and Installation

1.  **Clone the repository (if applicable) or download the files.**
2.  **Ensure Python 3.x is installed.**
3.  **Install required libraries:**
    It's recommended to use a virtual environment.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn chardet
    ```
    Alternatively, if a `requirements.txt` file is provided:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Dataset:**
    Place the `SampleSalesData.csv` file in the same directory as the Python script (`sales_analysis.py`).

## How to Run

1.  **Prepare the data:** Make sure `SampleSalesData.csv` is in the project's root directory.
2.  **Execute the script:**
    Rename `sales_analysis.txt` to `sales_analysis.py` (if you haven't already). Then run:
    ```bash
    python sales_analysis.py
    ```
3.  **Output:**
    The script will:
    *   Print the Mean Absolute Error, Mean Squared Error, and R² Score for the model.
    *   Display two plots:
        *   Actual vs. Predicted Sales
        *   Residual Plot
    *   Print the predicted sales for a sample new data point defined within the script.

## Making New Predictions

To predict sales for a new data point:
1.  Locate the `NEW PREDICTION` section in the `sales_analysis.py` script.
2.  Modify the example values for `quantity_ordered`, `price_each`, `msrp`, `year`, `month`, `status`, `productline`, `productcode`, `territory`, and `dealsize` according to your new data.
    ```python
    # Example values for the new data point:
    quantity_ordered = 30  # Change this
    price_each = 50        # Change this
    # ... and so on for other features
    ```
3.  Re-run the script. The predicted sales for your new data will be printed at the end of the output.

    **Important:** Ensure that the categorical values (like `status`, `productline`, `territory`, `dealsize`, `productcode`) you provide for new predictions exist in the original training data. If a new category is introduced that the model hasn't seen, its corresponding one-hot encoded column won't exist, and the `create_new_data` function will assign 0 to it, effectively treating it as an unknown or base category (if `drop_first=True` was used and it was the first category).

## Future Work (Potential Improvements)

*   Implement K-Fold Cross-Validation for more robust model evaluation (imports are present but not used).
*   Perform hyperparameter tuning for the Random Forest Regressor (e.g., using GridSearchCV or RandomizedSearchCV).
*   Explore other regression models (e.g., Gradient Boosting, XGBoost, Linear Regression).
*   Conduct more in-depth Exploratory Data Analysis (EDA) to uncover more insights.
*   Develop a simple web interface (e.g., using Flask or Streamlit) for easier interaction with the prediction model.
