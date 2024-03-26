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
#nothing

class NewsTextSpider(scrapy.Spider):
    name = 'news_text'

    def start_requests(self):
        # Get the list of tickers from the stock data files
        tickers = [filename.split('_')[0] for filename in os.listdir('data') if filename.endswith('_stock_data.txt')]

        for ticker in tickers:
            news_file_path = f'data/{ticker}_news.txt'
            if os.path.exists(news_file_path):
                with open(news_file_path, 'r') as f:
                    for line in f:
                        if line.startswith('Link:'):
                            link = line.strip('Link: ').strip()
                            yield scrapy.Request(url=link, callback=self.parse_article, meta={'ticker': ticker})

    def parse_article(self, response):
        ticker = response.meta['ticker']
        #jj

        # Extract the article text using appropriate selectors
        article_text = response.css('p::text').getall()  # Adjust selector as needed

        # Save the article text to a file
        filename = f'data/articles/{ticker}_article_{response.url.split("/")[-2]}.txt'
        with open(filename, 'w') as f:
            f.write(' '.join(article_text))