import csv
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from pymongo import MongoClient
import os
import json
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
import pymongo
from pymongo.errors import PyMongoError
from connectionstrings import mongo_connection_string, database_name, collection_name

from pymongo.errors import PyMongoError
#\smi_scraper\downloads\mongofin2.py
data_directory = "./data"

# Ensure all necessary directories exist
os.makedirs(data_directory, exist_ok=True)

def fetch_summary_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    # Selecting a comprehensive set of metrics from the info dictionary
    key_metrics = {key: info[key] for key in ['previousClose', 'open', 'bid', 'ask', 'marketCap',
                                              'beta', 'trailingPE', 'volume', 'averageVolume', 
                                              'dividendYield', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 
                                              'sector', 'fullTimeEmployees', 'longBusinessSummary']}
    return key_metrics

def retrieve_data(ticker, period="1y", interval="1d"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    if not hist.empty:
        # Including the latest news in the data retrieval
        news_data = stock.news
        summary_data = fetch_summary_data(ticker)
        return {
            **summary_data,
            'News': news_data,
            'HistoricalData': hist.iloc[0].to_dict()
        }
    else:
        print(f"{ticker}: No data found.")
        return None

def save_to_mongo(ticker, data):
    client = pymongo.MongoClient(mongo_connection_string)
    db = client[database_name]
    collection = db[collection_name]
    try:
        # Ensure compatibility with MongoDB by converting numpy data types
        data = json.loads(json.dumps(data, default=lambda x: str(x)))
        document = {'ticker': ticker, 'data': data, 'timestamp': datetime.datetime.now()}
        collection.insert_one(document)
        print(f"Data for {ticker} saved to MongoDB.")
    except PyMongoError as e:
        print(f"Failed to save data for {ticker} to MongoDB: {e}")

def save_response_to_file(ticker, data):
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
        if filename.startswith(ticker):
            filepath = os.path.join(data_directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                accumulated_data.append(data)
    return accumulated_data

if __name__ == "__main__":
    # Define the path to the ticker file
    ticker_filepath = os.path.join(data_directory, r'C:\Users\jucke\Desktop\Juckesam\projectjuckes\data\Ticker\SP500.csv')
    if not os.path.exists(ticker_filepath):
        print(f"Ticker file not found at {ticker_filepath}")
    else:
        ticker_df = pd.read_csv(ticker_filepath)
        tickers = ticker_df['Symbol'].tolist()

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
