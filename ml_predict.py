import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Fetch Stock Price Data
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
data = yf.download(tickers, start="2020-01-01", end="2021-01-01")['Adj Close']

# Step 2: Create Correlation Matrix
correlation_matrix = data.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# For the machine learning model, let's predict AAPL prices based on other stock prices.
# This is just a simple illustrative example and not a practical trading strategy.

# Step 3: Prepare Data for ML Model
X = data.drop('AAPL', axis=1)
y = data['AAPL']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and Train the Machine Learning Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions and Evaluate the Model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("Root Mean Square Error (RMSE):", rmse)

# Plotting and further analysis can be done as needed.
