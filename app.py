from flask import Flask, render_template, jsonify  # Import render_template
import os
import pymongo
from connection_strings import mongo_connection_string


# Connect to MongoDB
client = pymongo.MongoClient(mongo_connection_string)
db = client.your_database_name  # Replace with your database name
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/news_data/<ticker>')
def get_news_data(ticker):
    articles_dir = 'C:/Users/jucke/Desktop/Juckesam/projectjuckes/smi_scraper/smi_scraper/spiders/data/articles'  # Replace with the actual path
    article_files = [f for f in os.listdir(articles_dir) if f.startswith(ticker + '_article_')]

    news_data = []
    for article_file in article_files:
        with open(os.path.join(articles_dir, article_file), 'r') as f:
            article_text = f.read()
            news_data.append({'text': article_text})

    return jsonify(news_data)

if __name__ == '__main__':
    app.run(debug=True)