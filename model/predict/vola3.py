import yfinance as yf
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Function to fetch historical data
def fetch_data(ticker):
    data = yf.download(ticker, period="2y")
    return data['Adj Close']

# Function to calculate volatility
def calculate_volatility(data):
    log_returns = np.log(data / data.shift(1))
    volatility = log_returns.rolling(window=7).std().dropna()
    return volatility

# Function to load news sentiment
def load_news_sentiment(filepath, ticker):
    with open(filepath, 'r') as file:
        for line in file:
            article = json.loads(line)
            if article['ticker'] == ticker:
                return article['modelResultIsPositive']
    return 0  # Default sentiment if not found

# Prepare dataset for model
def prepare_dataset(tickers, sentiment_file):
    dataset = pd.DataFrame()
    for ticker in tickers:
        print(f'Processing {ticker}')
        data = fetch_data(ticker)
        volatility = calculate_volatility(data)
        sentiment = load_news_sentiment(sentiment_file, ticker)
        temp_df = pd.DataFrame({
            'Volatility': volatility,
            'Sentiment': [sentiment] * len(volatility)  # Repeat sentiment value for the length of volatility data
        }, index=volatility.index)
        dataset = pd.concat([dataset, temp_df])
    return dataset.dropna()

# Example Machine Learning Application
def apply_ml_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    return model, mse, r2

# Load the list of tickers from a file or define it directly
tickers = ['AAPL', 'MSFT']  # Example, replace with your actual list of tickers
sentiment_file = 'C:/Users/jucke/Desktop/Juckesam/projectjuckes/model/Semantic/datasemantic/processed_articles.jl'
# dataset = prepare_dataset(sp500_tickers[:10], sentiment_file)  # Limited to first 10 for brevity

# Prepare the dataset
dataset = prepare_dataset(tickers, sentiment_file)
X = dataset[['Sentiment']]
y = dataset['Volatility']

# Apply machine learning model
model, mse, r2 = apply_ml_model(X, y)
