# Boston House Price Predictor 🏡🤖

A simple yet effective Machine Learning project built to understand the core concepts of **Simple Linear Regression**. This model predicts house prices based on the average number of rooms using the classic Boston Housing dataset.

## 🚀 Technical Stack
* **Language:** Python
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Seaborn, Matplotlib

## 🧠 Model Workflow
1. **Data Preprocessing:** Fetched raw dataset and handled missing values using `dropna()`.
2. **Feature Engineering:** Isolated the `rm` (rooms) column as the independent variable (X) and `medv` (price) as the target variable (y).
3. **Training:** Split the data (80/20) and trained a Scikit-Learn `LinearRegression` model.
4. **Evaluation:** Evaluated model performance using Mean Squared Error (MSE) and R-squared (R2) metrics.
5. **Visualization:** Plotted the best-fit regression line using Seaborn.

## 📈 Example Output
* **Input:** 6 Rooms
* **Predicted Price:** ~$19,840 

*Developed by Anshika Batham | B.Tech Artificial Intelligence*
