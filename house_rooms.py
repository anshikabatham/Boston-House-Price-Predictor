"""
Boston House Price Prediction (Single Variable Linear Regression)
Author: Anshika Batham
Description: A simple machine learning model to predict house prices based on the number of rooms using the Boston Housing dataset.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # 1. Load and Clean Data
    print("[INFO] Fetching dataset...")
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    df = df.dropna()

    # 2. Feature Selection
    # X: Number of Rooms (rm), y: Median Value of Owner-Occupied Homes (medv)
    X = df[['rm']]
    y = df['medv']

    # 3. Train-Test Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Model Initialization and Training
    print("[INFO] Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Model Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2) Score:     {r2:.2f}")

    # 6. Real-World Prediction Test
    test_rooms = 6
    predicted_price = model.predict([[test_rooms]])
    print("\n--- Sample Prediction ---")
    print(f"Input: {test_rooms} Rooms")
    # Note: 'medv' in Boston dataset is in $1000s
    print(f"Predicted House Price:    ${predicted_price[0] * 1000:,.2f}") 

    # 7. Data Visualization
    print("\n[INFO] Generating regression plot. Close the window to exit.")
    plt.figure(figsize=(8, 5))
    sns.regplot(x=X_test['rm'], y=y_test, scatter_kws={'color': '#005b96', 'alpha':0.6}, line_kws={'color': '#ff0000'})
    plt.title("Linear Regression: House Price vs Number of Rooms")
    plt.xlabel("Average Number of Rooms (RM)")
    plt.ylabel("House Price (MEDV in $1000s)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()
    