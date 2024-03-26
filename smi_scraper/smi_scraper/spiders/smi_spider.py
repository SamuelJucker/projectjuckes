import scrapy
from scrapy.selector import Selector
from scrapy_selenium import SeleniumRequest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
import requests  # Added for get_page function
from bs4 import BeautifulSoup  # Added for get_page function
import pandas as pd  # Added for data manipulation
import yfinance as yf  # Import yfinance library

class SmiSpider(scrapy.Spider):
    name = 'smi'

    # Define your API key as a variable
    
    api_key = 'bzffsumg9ygmd0nx8am7ubd70n0vbsq3slofzhsc'

    def start_requests(self):
        # Get the list of SMI tickers (replace with your actual method)
        smi_tickers = ['ABBN', 'ADEN', 'ALC', 'CSGN', 'GEBN', 'GIVN', 'BAER', 'LHN', 'LONN', 'NESN', 'NOVN', 'PGHN', 'CFR', 'ROG', 'SLHN', 'SREN', 'SCMN', 'SGSN', 'UBSG', 'ZURN','AAPL', 'UHR', 'TSLA','AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V', 'WBA', 'WMT', 'DIS']

        for ticker in smi_tickers:
            stock_data_url = f'https://finance.yahoo.com/quote/{ticker}'
            news_api_url = f'https://stocknewsapi.com/api/v1?tickers={ticker}&items=3&page=1&token={self.api_key}'

            options = Options()
            options.add_argument('--headless')  # Run in headless mode
            driver = webdriver.Chrome(options=options)

            yield SeleniumRequest(
                url=stock_data_url,
                callback=self.parse_stock_data,
                meta={'ticker': ticker, 'driver': driver}
            )

            yield scrapy.Request(
                url=news_api_url,
                callback=self.parse_news_api,
                meta={'ticker': ticker}
            )

    def parse_stock_data(self, response):
        driver = response.meta['driver']
        ticker = response.meta['ticker']

        # Wait for dynamic content to load
        time.sleep(5)  # Adjust wait time as needed

        # Extract stock data using appropriate selectors
        stock_data = {
            'price': response.css('span[data-reactid="32"]::text').get(),
            # ... (extract other relevant data using CSS selectors)
        }

        # Retrieve important information from yfinance API
        yf_data = yf.Ticker(ticker).info
        important_info = {
            'market_cap': yf_data.get('marketCap'),
            'dividend_yield': yf_data.get('dividendYield'),
            # ... (add other important information you want to extract)
        }

        # Combine stock data and yfinance information
        combined_data = {**stock_data, **important_info}

        # Save data to text file
        self.save_to_text_file(ticker, combined_data, 'stock_data')

        driver.quit()

    def parse_news_api(self, response):
        ticker = response.meta['ticker']
        news_data = response.json()

        # Extract news articles from the API response
        news_articles = []
        for article in news_data['data']:
            news_articles.append({
                'title': article['title'],
                'link': article['news_url'],
                'timestamp': article['date']
            })

        # Save news data to text file
        self.save_to_text_file(ticker, news_articles, 'news')

    def save_to_text_file(self, ticker, data, data_type):
        # Create data folder if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')

        # Construct file path
        file_path = f'data/{ticker}_{data_type}.txt'

        # Write data to file
        with open(file_path, 'w') as f:
            f.write(f"Ticker: {ticker}\n")
            if data_type == 'news':
                for article in data:
                    f.write(f"Title: {article['title']}\n")
                    f.write(f"Link:              {article['link']}\n")
                    f.write(f"Timestamp: {article['timestamp']}\n\n")
            else:
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")

    # Added function for downloading and parsing web pages
    def get_page(self, url):
        """Download a webpage and return a BeautifulSoup doc"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        if not response.ok:
            print('Status code:', response.status_code)
            raise requests.exceptions.RequestException('Failed to load page {}'.format(url))
        doc = BeautifulSoup(response.text, 'html.parser')
        return doc