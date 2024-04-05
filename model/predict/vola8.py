import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
from datetime import datetime

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Create Model')
parser.add_argument('-u', '--uri', required=True, help="mongodb uri with username/password")
args = parser.parse_args()

# Connect to MongoDB
client = MongoClient(args.uri)
db = client["juckesamDB"]
collection = db["juckesamCollection"]

# Load the ticker list from CSV
ticker_df = pd.read_csv('C:/Users/jucke/Desktop/Juckesam/projectjuckes/data/Ticker/SP500.csv')
ticker_symbols = ticker_df['Symbol'].tolist()
selected_symbols = ticker_symbols[:60]

# Function to fetch historical data

def fetch_data(ticker):
    try:
        data = yf.download(ticker, start="2024-01-01", end="2024-04-05")
        return data['Adj Close']
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")
        return None

def calculate_volatility(data):
    log_returns = np.log(data / data.shift(1))
    volatility = log_returns.rolling(window=7).std().dropna()
    return volatility


def load_and_aggregate_sentiment(filepath, tickers):
    data_list = []
    with open(filepath, 'r') as file:
        for line in file:
            article = json.loads(line)
            if article['ticker'] in tickers:
                date = datetime.strptime(article['providerPublishTime'], "%Y-%m-%d %H:%M:%S").date()
                data_list.append({'Date': date, 'Ticker': article['ticker'], 'Sentiment': article['modelResultIsPositive']})

    sentiment_data = pd.DataFrame(data_list)
    if not sentiment_data.empty:
        sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])
        aggregated_sentiment = sentiment_data.groupby(['Ticker', 'Date'], as_index=False).mean()
        return aggregated_sentiment
    return pd.DataFrame()



def prepare_dataset(tickers, sentiment_file):
    sentiment_data = load_and_aggregate_sentiment(sentiment_file, tickers)
    dataset = pd.DataFrame()
    for ticker in tickers:
        print(f'Processing {ticker}')
        data = fetch_data(ticker)
        if data is None:  # Skip tickers for which data couldn't be fetched
            continue
        
         # Calculate volatility and ensure 'Date' is the index in datetime format
        volatility = calculate_volatility(data)
        volatility_df = pd.DataFrame(data=volatility, columns=['Volatility'])  # Directly naming the column 'Volatility'
        volatility_df['Date'] = volatility_df.index
        volatility_df.reset_index(drop=True, inplace=True)
        volatility_df['Date'] = pd.to_datetime(volatility_df['Date'])

        # Load and prepare sentiment data
        ticker_sentiment = sentiment_data[sentiment_data['Ticker'] == ticker]
        ticker_sentiment.loc[:, 'Date'] = pd.to_datetime(ticker_sentiment['Date'])

        # Merge on 'Date'
        merged_data = pd.merge(volatility_df, ticker_sentiment, how='left', on='Date')
        merged_data.set_index('Date', inplace=True)

        # Handle missing sentiment values
        merged_data['Sentiment'] = merged_data['Sentiment'].fillna(0)  # Assuming 0 for missing sentiment

        # Append to the dataset
        dataset = pd.concat([dataset, merged_data[['Volatility', 'Sentiment']]])

    return dataset.dropna()
   

# Assuming machine learning model application remains unchanged


# Apply machine learning model
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



if __name__ == '__main__':
    sentiment_file = 'C:/Users/jucke/Desktop/Juckesam/projectjuckes/model/Semantic/datasemantic/processed_articles.jl'
    dataset = prepare_dataset(selected_symbols, sentiment_file)
    
    if not dataset.empty:
        X = dataset[['Sentiment']]
        y = dataset['Volatility']
        model, mse, r2 = apply_ml_model(X, y)
        model_file_path = r'C:\Users\jucke\Desktop\Juckesam\projectjuckes\model\predict\model_filevola.pkl'
        with open(model_file_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {model_file_path}")
    else:
        print("Dataset is empty. Check data availability and filters.")