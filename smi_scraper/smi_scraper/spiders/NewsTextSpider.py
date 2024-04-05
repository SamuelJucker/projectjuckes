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
from smi_scraper.connection_strings import mongo_connection_string,  database_name, collection_name
import datetime
import json

from smi_scraper.items import NewsArticleItem  # Ensure you have this item defined

from pymongo import MongoClient

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

# Define your item (in items.py)
class NewsArticleItem(scrapy.Item):
    ticker = scrapy.Field()
    providerPublishTime = scrapy.Field()
    articleText = scrapy.Field()

# Define your MongoDB pipeline (in pipelines.py)
class MongoDBPipeline(object):
    def open_spider(self, spider):
        self.client = MongoClient(mongo_connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

    def close_spider(self, spider):
        self.client.close()

    def process_item(self, item, spider):
        self.collection.insert_one(dict(item))
        return item

# # ...

# class NewsTextSpider(scrapy.Spider):
#     name = 'news_text'
#     article_limit = 10  # Set the limit of articles to scrape
#     articles_processed = 0  # Initialize counter for processed articles
#     def start_requests(self):
#         news_data = self.read_news_data()
#         for item in news_data:
#             if self.articles_processed < self.article_limit:
#                 yield scrapy.Request(url=item['link'], callback=self.parse_article, meta={'ticker': item['ticker'], 'providerPublishTime': item['providerPublishTime']})
#             else:
#                 break  # 
#     def parse_article(self, response):
#         self.articles_processed += 1  # Increment the counter for each processed article
#         ticker = response.meta['ticker']
#         provider_publish_time = response.meta['providerPublishTime']

#         # Extract the article text using appropriate selectors
#         provider_publish_time_safe = provider_publish_time.replace(':', '_')

#     # Extract the article text using appropriate selectors
#         article_text = response.css('p::text').getall()
#         filename = f'C:/Users/jucke/Desktop/Juckesam/projectjuckes/data/articles/{ticker}_article_{provider_publish_time_safe}.txt'
#         os.makedirs(os.path.dirname(filename), exist_ok=True)
#         with open(filename, 'w', encoding='utf-8') as f:
#             f.write(' '.join(article_text))
#         self.log(f'Saved file {filename}')

#     def read_news_data(self):
#         data_folder = 'C:/Users/jucke/Desktop/Juckesam/projectjuckes/data'
#         news_data = []

#         # List all .json files in the data directory
#         for filename in os.listdir(data_folder):
#             if filename.endswith('.json'):
#                 with open(os.path.join(data_folder, filename), 'r', encoding='utf-8') as file:
#                     json_data = json.load(file)
#                     ticker = filename.split('_')[0]  # Extract ticker from filename

#                     # Check if 'News' field is in the JSON data
#                     if 'News' in json_data:
#                         for news_item in json_data['News']:
#                             # Check if 'providerPublishTime' is in the news item
#                             if 'providerPublishTime' in news_item:
#                                 # Convert UNIX timestamp to datetime
#                                 provider_time = datetime.datetime.fromtimestamp(news_item['providerPublishTime'])
#                                 formatted_time = provider_time.strftime('%Y-%m-%d %H:%M:%S')

#                                 # Store the ticker, link, and formatted time in a list
#                                 news_data.append({
#                                     'ticker': ticker,
#                                     'link': news_item['link'],
#                                     'providerPublishTime': formatted_time
#                                 })

#         return news_data


# # The rest of your spider code...



#         # Extract the article text using appropriate selectors
#         article_text = response.css('p::text').getall()  # Adjust selector as needed

#         # Save the article text to a file
#         filename = f'data/articles/{ticker}_article_{response.url.split("/")[-2]}.txt'
#         with open(filename, 'w') as f:
#             f.write(' '.join(article_text))