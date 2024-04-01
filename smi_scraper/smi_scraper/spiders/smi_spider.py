import scrapy
from scrapy.selector import Selector
from scrapy_selenium import SeleniumRequest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
import datetime
import requests  # Added for get_page function
from bs4 import BeautifulSoup  # Added for get_page function
import pandas as pd  # Added for data manipulation
import yfinance as yf  # Import yfinance library
import pymongo
import json
from smi_scraper.connection_strings import mongo_connection_string, database_name, collection_name





class StockSpider(scrapy.Spider):
    name = 'stock'

    def start_requests(self):
        tickers = ['ABBN', 'ALC', 'CSGN', 'AAPL', 'IBM', 'GEBN', 'GIVN'][:7]
        for ticker in tickers:
            try:
                data = self.retrieve_data(ticker)
                if data:
                    self.save_to_mongo(ticker, data, 'yfinance_data')
                    self.save_response_to_file(data, ticker)
                    self.print_data(ticker, data)
                else:
                    print(f"No data available for {ticker}.")
            except Exception as e:
                print(f"Failed to process {ticker}: {e}")

    def retrieve_data(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.history(period="1d")
            if not info.empty:
                # Convert numpy types to Python native types for BSON serialization
                return {
                    'Open': float(info['Open'][0]),
                    'High': float(info['High'][0]),
                    'Low': float(info['Low'][0]),
                    'Close': float(info['Close'][0]),
                    'Volume': int(info['Volume'][0])  # Convert Volume to int
                }
            else:
                return None
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def save_to_mongo(self, ticker, data, data_type):
        try:
            client = pymongo.MongoClient(mongo_connection_string)
            db = client[database_name]
            collection = db[collection_name]
            document = {'ticker': ticker, 'data_type': data_type, 'data': data}
            collection.insert_one(document)
        except Exception as e:
            print(f"Error saving {ticker} to MongoDB: {e}")

    def save_response_to_file(self, data, ticker):
        try:
            now = datetime.datetime.now().strftime("%Y%m%d%H%M")
            filename = f"data/MD_{now}_{ticker}.txt"
            os.makedirs('data', exist_ok=True)
            with open(filename, 'w') as f:
                f.write(json.dumps(data, indent=4))
        except Exception as e:
            print(f"Error saving {ticker} data to file: {e}")

    def print_data(self, ticker, data):
        print(f"Ticker: {ticker}")
        for key, value in data.items():
            print(f"{key}: {value}")
        print("-" * 50)

# class StockSpider(scrapy.Spider):
#     name = 'stock'

#     def start_requests(self):
#         # Limit the tickers to the first 5
#         tickers = ['ABBN', 'ALC', 'CSGN', 'AAPL','IBM','GEBN', 'GIVN'][:7]
#         for ticker in tickers:
#             # Retrieve data from yfinance API
#             data = self.retrieve_data(ticker)
#             if data:
#                 # Save the data to MongoDB
#                 self.save_to_mongo(ticker, data, 'yfinance_data')
#                 # Save the data to a text file
#                 self.save_response_to_file(data, ticker)
#                 # Print the data
#                 self.print_data(ticker, data)
#             else:
#                 print(f"No data available for {ticker}.")

#     def retrieve_data(self, ticker):
#         try:
#             stock = yf.Ticker(ticker)
#             info = stock.history(period="1d")
#             if not info.empty:
#                 return {
#                     'Open': info['Open'][0],
#                     'High': info['High'][0],
#                     'Low': info['Low'][0],
#                     'Close': info['Close'][0],
#                     'Volume': info['Volume'][0]
#                 }
#             else:
#                 return None
#         except Exception as e:
#             print(f"Error occurred while fetching data for {ticker}: {e}")
#             return None

#     def save_to_mongo(self, ticker, data, data_type):
#         client = pymongo.MongoClient(mongo_connection_string)
#         db = client[database_name]
#         collection = db[collection_name]

#         # Prepare and insert the document
#         document = {
#             'ticker': ticker,
#             'data_type': data_type,
#             'data': data
#         }
#         collection.insert_one(document)

#     def save_response_to_file(self, data, ticker):
#         now = datetime.datetime.now().strftime("%Y%m%d%H%M")
#         filename = f"data/MD_{now}_{ticker}.txt"

#         # Ensure the data directory exists
#         os.makedirs('data', exist_ok=True)

#         # Save the data to the file
#         with open(filename, 'w') as f:
#             # Assuming 'data' is a dictionary, use json.dumps for better formatting
#             f.write(json.dumps(data, indent=4))

#     def print_data(self, ticker, data):
#         print(f"Ticker: {ticker}")
#         print(f"Open: {data['Open']}")
#         print(f"High: {data['High']}")
#         print(f"Low: {data['Low']}")
#         print(f"Close: {data['Close']}")
#         print(f"Volume: {data['Volume']}")
#         print("-" * 50)
        
        
    #     for ticker in smi_tickers:
    #         # Retrieve data from yfinance API
    #         yf_data = yf.Ticker(ticker).info
    #         # Save the essential yfinance information
    #         self.save_to_mongo(ticker, yf_data, 'yfinance_data')
    #         # Save the yfinance response to a file
    #         self.save_response_to_file(yf_data, ticker)

    # def save_to_mongo(self, ticker, data, data_type):
  
    #     print(mongo_connection_string)  # For debugging purposes only. Remove after use.


    #     client = pymongo.MongoClient(mongo_connection_string)
    #     db = client[database_name]
    #     collection = db[collection_name]
        
    #     # Prepare and insert the document
    #     document = {
    #         'ticker': ticker,
    #         'data_type': data_type,
    #         'data': data
    #     }
    #     collection.insert_one(document)

    # def save_response_to_file(self, data, ticker):
    #     now = datetime.datetime.now().strftime("%Y%m%d%H%M")
    #     filename = f"data/MD_{now}_{ticker}.txt"
        
    #     # Ensure the data directory exists
    #     os.makedirs('data', exist_ok=True)
        
    #     # Save the data to the file
    #     with open(filename, 'w') as f:
    #         # Assuming 'data' is a dictionary, use json.dumps for better formatting
    #         f.write(json.dumps(data, indent=4))

    # You can remove the get_page method if you are not using it

# class StockSpider(scrapy.Spider):
#     name = 'stock'

#     # Define your API key as a variable
#     api_key = 'bzffsumg9ygmd0nx8am7ubd70n0vbsq3slofzhsc'

#     def start_requests(self):
#         # Get the list of SMI tickers
#         # smi_tickers = ['ABBN', 'ADEN', 'ALC', 'CSGN', 'GEBN', 'GIVN', 'BAER', 'LHN', 'LONN', 'NESN', 'NOVN', 'PGHN', 'CFR', 'ROG', 'SLHN', 'SREN', 'SCMN', 'SGSN', 'UBSG', 'ZURN', 'AAPL', 'UHR', 'TSLA', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V', 'WBA', 'WMT', 'DIS']
#         smi_tickers = ['ABBN', 'ALC', 'CSGN', 'GEBN', 'GIVN', 'NESN', 'NOVN',  'SLHN', 'SREN', 'SCMN', 'SGSN',  'ZURN', 'AAPL',   'CVX', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'KO', 'JPM', 'MSFT']

#         for ticker in smi_tickers:
#             # Retrieve data from yfinance API
#             yf_data = yf.Ticker(ticker).info

#             # Always save the essential yfinance information first
#             self.save_to_mongo(ticker, yf_data, 'yfinance_data')
#             self.save_response_to_file(yf_data, ticker) 
#             # URLs for additional data
#             stock_data_url = f'https://finance.yahoo.com/quote/{ticker}'
#             news_api_url = f'https://stocknewsapi.com/api/v1?tickers={ticker}&items=3&page=1&token={self.api_key}'

#             # Proceed with additional requests
#             yield scrapy.Request(url=stock_data_url, callback=self.parse_stock_data, meta={'ticker': ticker, 'yf_data': yf_data})
#             yield scrapy.Request(url=news_api_url, callback=self.parse_news_api, meta={'ticker': ticker})

#     def parse_stock_data(self, response):
#         ticker = response.meta['ticker']
#         yf_data = response.meta['yf_data']

#         # Extract additional stock data here if necessary
#         additional_stock_data = {
#             'price': response.css('fin-streamer[data-symbol="{}"]::text'.format(ticker)).get(),
#             # Extract other relevant data using CSS selectors
#         }

#         # Combine yfinance information with additional stock data
#         combined_data = {**yf_data, **additional_stock_data}

#         # Save combined data to MongoDB
#         self.save_to_mongo(ticker, combined_data, 'combined_stock_data')

#     def parse_news_api(self, response):
#         ticker = response.meta['ticker']
#         news_data = response.json()

#         # Extract and save news articles from the API response
#         news_articles = [
#             {
#                 'title': article['title'],
#                 'link': article['news_url'],
#                 'timestamp': article['date']
#             } for article in news_data.get('data', [])
#         ]
#         self.save_to_mongo(ticker, news_articles, 'news')
        
        
        
        
#     def save_to_mongo(self, ticker, data, data_type):
#         client = pymongo.MongoClient(mongo_connection_string)
#         db = client[database_name]
#         collection = db[collection_name]
        
#         # Prepare and insert the document
#         document = {
#             'ticker': ticker,
#             'data_type': data_type,
#             'data': data
#         }
#         collection.insert_one(document)


#         # Insert data into MongoDB collection


#     # ... (rest of the code remains the same)
#         # Save news data to text file
   
#     def save_response_to_file(self, data, ticker):
#         now = datetime.datetime.now().strftime("%Y%m%d%H%M")
#         filename = f"data/MD_{now}_{ticker}.txt"
        
#         # Ensure the data directory exists
#         os.makedirs(os.path.dirname(filename), exist_ok=True)
        
#         # Save the data to the file
#         with open(filename, 'w') as f:
#             f.write(str(data))
            
#     def save_to_text_file(self, ticker, data, data_type):
#         # Create data folder if it doesn't exist
#         if not os.path.exists('data'):
#             os.makedirs('data')

#         # Construct file path
#         file_path = f'data/{ticker}_{data_type}.txt'

#         # Write data to file
#         with open(file_path, 'w') as f:
#             f.write(f"Ticker: {ticker}\n")
#             if data_type == 'news':
#                 for article in data:
#                     f.write(f"Title: {article['title']}\n")
#                     f.write(f"Link:              {article['link']}\n")
#                     f.write(f"Timestamp: {article['timestamp']}\n\n")
#             else:
#                 for key, value in data.items():
#                     f.write(f"{key}: {value}\n")

#     # Added function for downloading and parsing web pages
#     def get_page(self, url):
#         """Download a webpage and return a BeautifulSoup doc"""
#         headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
#         }
#         response = requests.get(url, headers=headers)
#         if not response.ok:
#             print('Status code:', response.status_code)
#             raise requests.exceptions.RequestException('Failed to load page {}'.format(url))
#         doc = BeautifulSoup(response.text, 'html.parser')
#         return doc