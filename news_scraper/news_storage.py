import pymongo

# adjust these as needed
client = pymongo.MongoClient('localhost', 27017)
db = client['your_database_name']
news_collection = db['your_collection_name']

def save_news(item):
    news_collection.insert_one(item.asdict())
