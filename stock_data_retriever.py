import os
import json
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
import pymongo
from pymongo.errors import PyMongoError
import argparse
# from connectionstrings import mongo_connection_string, database_name, collection_name

data_directory = "./data"

database_name = os.getenv('DATABASE_NAME', 'juckesamDB')
collection_name = os.getenv('COLLECTION_NAME', 'juckesamCollection')

def retrieve_data(ticker, period="1y", interval="1d"):
    stock = yf.Ticker(ticker)
    info = stock.history(period=period, interval=interval)
    
    
    if not info.empty:
        news_data = stock.news  
        data = {
            'Open': float(info['Open'].iloc[0]),
            'High': float(info['High'].iloc[0]),
            'Low': float(info['Low'].iloc[0]),
            'Close': float(info['Close'].iloc[0]),
            'Volume': int(info['Volume'].iloc[0]),
            'Adj Close': float(info['Adj Close'].iloc[0]) if 'Adj Close' in info else None,
            'News': news_data  # Including the fetched news data
        }
        return data
    else:
        print(f"{ticker}: No data found.")
        return None

def save_to_mongo(ticker, data):
    try:
        client = pymongo.MongoClient(mongo_connection_string)
        db = client[database_name]
        collection = db[collection_name]

        # Convert numpy types to Python types for MongoDB
        for key, value in data.items():
            if isinstance(value, (np.int64, np.float64, np.ndarray)):
                data[key] = value.item() if value.size == 1 else value.tolist()

        document = {
            'ticker': ticker,
            'data': data,
            'timestamp': datetime.datetime.now()
        }

        collection.insert_one(document)
        print(f"Data for {ticker} saved to MongoDB.")
    except PyMongoError as e:
        print(f"Failed to save data for {ticker} to MongoDB: {e}")



def save_response_to_file(ticker, data):
    os.makedirs(data_directory, exist_ok=True)
    filename = os.path.join(data_directory, f"{ticker}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.json")
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data for {ticker} saved to file: {filename}")

def print_data(ticker, data):
    print(f"Ticker: {ticker}")
    for key, value in data.items():
        print(f"{key}: {value}")
    print("-" * 50)

def accumulate_data_from_files(ticker):
    accumulated_data = []
    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        if os.path.isfile(filepath) and filename.startswith(ticker):
            with open(filepath, 'r') as file:
                data = json.load(file)
                accumulated_data.append(data)
    return accumulated_data

if __name__ == "__main__":
    # Use argparse to parse the MongoDB URI from the command line
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-u', '--mongo_uri', type=str, default=os.getenv('MONGO_URI', 'your_default_mongo_uri'), help='MongoDB URI string')
    args = parser.parse_args()

    # mongo_connection_string = args.uri
    mongo_connection_string = args.mongo_uri

    # ticker_filepath = os.path.join(data_directory, 'Ticker/SP500.csv')
    ticker_filepath = 'SP500.csv'

    ticker_df = pd.read_csv(ticker_filepath)
    tickers = ticker_df['Symbol'].tolist()[:500]
    
    for ticker in tickers:
        data = retrieve_data(ticker)
        if data:
            save_to_mongo(ticker, data)
            save_response_to_file(ticker, data)
            print_data(ticker, data)

            accumulated_data = accumulate_data_from_files(ticker)
            if accumulated_data:
                print(f"Accumulated data for {ticker}:")
                print(accumulated_data)
            else:
                print(f"No accumulated data found for {ticker}.")
