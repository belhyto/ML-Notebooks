# Importing necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = '/kaggle/input/life-expectancy-who/Life Expectancy Data.csv'
data = pd.read_csv(dataset_path)

# Preprocessing: Drop rows with missing values and select relevant features
data.dropna(inplace=True)
# Preprocessing: Selecting features and target variable
X = data[['GDP']]  # Selecting 'GDP' as the feature
y = data['Life expectancy ']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = regressor.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print("Y-intercept:", regressor.intercept_)
print("Slope:", regressor.coef_)

# Scatter plot of the actual data
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data',alpha=0.5)

# Plot the best-fit line
plt.plot(X_test, y_pred, color='black', linewidth=2, label='Best Fit Line')

plt.title('Scatter Plot with Best Fit Line')
plt.xlabel('GDP')
plt.ylabel('Life Expectancy')
plt.legend()
plt.grid(True)
plt.show()
