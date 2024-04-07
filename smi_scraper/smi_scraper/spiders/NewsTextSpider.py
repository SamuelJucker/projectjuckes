import scrapy
from scrapy.selector import Selector
from scrapy_selenium import SeleniumRequest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
import sys
import requests  # Added for get_page function
from bs4 import BeautifulSoup  # Added for get_page function
import pandas as pd  # Added for data manipulation
#nothing
import pymongo
# from smi_scraper.connection_strings import mongo_connection_string,  database_name, collection_name
import datetime
import json
import argparse

from smi_scraper.items import NewsArticleItem  # Ensure you have this item defined

from pymongo import MongoClient


with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Set default values from the config file
mongo_uri = config.get('MONGO_URI', 'default_mongo_uri')
database_name = config.get('DATABASE_NAME', 'default_database_name')
collection_name = config.get('COLLECTION_NAME', 'default_collection_name')
class NewsTextSpider(scrapy.Spider):
    name = 'news_text'
    article_limit = 99999
    custom_settings = {
        'ITEM_PIPELINES': {
            'smi_scraper.pipelines.MongoDBPipeline': 300,
        }
    }

    def start_requests(self):
        news_data = self.read_news_data()
        for item in news_data[:self.article_limit]:  # Assuming you have defined article_limit
            yield scrapy.Request(url=item['link'], callback=self.parse_article, meta=item)

    def parse_article(self, response):
        ticker = response.meta['ticker']
        provider_publish_time = response.meta['providerPublishTime']
        provider_publish_time_safe = provider_publish_time.replace(':', '_')

        article_text = ' '.join(response.css('p::text').getall())
        item = NewsArticleItem(ticker=ticker, providerPublishTime=provider_publish_time, articleText=article_text)

        yield item  # Yield the item for saving in MongoDB and .jl file

    def read_news_data(self):
        data_folder = 'C:/Users/jucke/Desktop/Juckesam/projectjuckes/data'
        news_data = []
        for filename in os.listdir(data_folder):
            if filename.endswith('.json'):
                with open(os.path.join(data_folder, filename), 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                    ticker = filename.split('_')[0]
                    if 'News' in json_data:
                        for news_item in json_data['News']:
                            if 'providerPublishTime' in news_item:
                                provider_time = datetime.datetime.fromtimestamp(news_item['providerPublishTime'])
                                formatted_time = provider_time.strftime('%Y-%m-%d %H:%M:%S')
                                news_data.append({
                                    'ticker': ticker,
                                    'link': news_item['link'],
                                    'providerPublishTime': formatted_time
                                })
        return news_data


if __name__ == "__main__":
    # Use argparse to parse the MongoDB URI from the command line
    parser = argparse.ArgumentParser(description='Run the news text spider')
    parser.add_argument('--mongo_uri', type=str, help='MongoDB URI string', default=mongo_uri)
    args = parser.parse_args()
    
    # Override the default mongo_uri if provided in the command line
    mongo_uri = args.mongo_uri