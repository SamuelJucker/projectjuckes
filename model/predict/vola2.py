import yfinance as yf
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go


def fetch_and_process_data(ticker):
    # Attempt to download stock data
    try:
        data = yf.download(ticker, period="2y")
        if data.empty:
            print(f"Skipping {ticker} as no data could be downloaded.")
            return None, None
        volatility = calculate_volatility(data['Adj Close'])
        return data, volatility
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None, None
    
def calculate_volatility(prices):
    log_returns = np.log(prices / prices.shift(1))
    volatility = log_returns.rolling(window=7).std().dropna()
    return volatility
# Function to fetch historical stock data
# def fetch_data(ticker):
#     data = yf.download(ticker, period="7d")
#     return data['Adj Close']

# # Function to calculate 7-day volatility
# def calculate_volatility(data):
#     log_returns = np.log(data / data.shift(1))
#     volatility = log_returns.rolling(window=7).std().dropna()
#     return volatility

# Function to load news sentiment from a .jl file
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
        data = fetch_data(ticker)
        volatility = calculate_volatility(data)
        sentiment = load_news_sentiment(sentiment_file, ticker)
        temp_df = pd.DataFrame({
            'Volatility': volatility,
            'Sentiment': sentiment
        })
        dataset = pd.concat([dataset, temp_df])
    return dataset.dropna()

# Train and evaluate the model
def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Plot results
    plot_results(y_test, predictions)
    
    return model, mse, r2

# Plot actual vs predicted volatility and sentiment vs volatility
def plot_results(y_test, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test, mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(y=predictions, mode='markers', name='Predicted'))
    fig.update_layout(title='Actual vs Predicted Volatility', xaxis_title='Index', yaxis_title='Volatility')
    fig.show()

# Load SP500 tickers
# sp500_tickers = pd.read_csv('C:/Users/jucke/Desktop/Juckesam/projectjuckes/data/Ticker/SP500.csv')['Ticker'].tolist()
sp500_tickers = pd.read_csv('C:/Users/jucke/Desktop/Juckesam/projectjuckes/data/Ticker/SP500.csv')['Symbol'].tolist()

# Prepare dataset
sentiment_file = 'C:/Users/jucke/Desktop/Juckesam/projectjuckes/model/Semantic/datasemantic/processed_articles.jl'
dataset = prepare_dataset(sp500_tickers[:10], sentiment_file)  # Limited to first 10 for brevity

X = dataset[['Sentiment']]
y = dataset['Volatility']

# Train and evaluate the model
model, mse, r2 = train_and_evaluate_model(X, y)
print(f'MSE: {mse}, R2: {r2}')
