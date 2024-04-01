# mongo_import.py

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pymongo import MongoClient
import gpxpy
import yfinance as yf
import gpxpy.gpx
from pathlib import Path


# Import database name and connection string from your connection strings file
from smi_scraper.connection_strings import database_name, collection_name, mongo_connection_string

# Function to create a MongoDB document from the yfinance data
def to_document(data):
    # Assuming data is a dictionary with the yfinance information
    return data

class YFinanceDataImporter:
    def __init__(self, mongo_uri, db=database_name, collection=collection_name):
        self.client = MongoClient(mongo_uri)
        self.db = db
        self.collection = collection

    def save_to_mongodb(self, data):
        db = self.client[self.db]
        collection = db[self.collection]
        document = to_document(data)
        collection.insert_one(document)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Import yfinance data to MongoDB")
    parser.add_argument('-u', '--uri', required=True, help="MongoDB URI with username/password")
    args = parser.parse_args()
    ticker = 'AAPL'
    yf_data = yf.Ticker(ticker).info

    # Initialize importer with command line arguments
    importer = YFinanceDataImporter(mongo_uri=args.uri)
    importer.save_to_mongodb(yf_data)
# Import database name and connection string from connectionstrings.py


class JsonLinesImporter:
    def __init__(self, file, mongo_uri, batch_size=30, db=database_name, collection=collection_name):
        self.file = file
        self.base_dir = Path(file).parent
        self.batch_size = batch_size
        self.client = MongoClient(mongo_uri)
        self.db = db
        self.collection = collection

    def read_lines(self):
        with open(self.file, 'r', encoding='UTF-8') as f:
            batch = []
            for line in f:
                batch.append(json.loads(line))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    def save_to_mongodb(self):
        db = self.client[self.db]
        collection = db[self.collection]
        for idx, batch in enumerate(self.read_lines()):
            print(f"Inserting batch {idx}")
            documents = self.prepare_documents(batch)
            if documents:
                collection.insert_many(documents)

    def prepare_documents(self, batch):
        with ProcessPoolExecutor() as executor:
            documents = list(executor.map(to_document, [self.base_dir] * len(batch), batch))
        return [doc for doc in documents if doc]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Import JSON Lines file to MongoDB")
    parser.add_argument('-u', '--uri', required=True, help="MongoDB URI with username/password")
    parser.add_argument('-i', '--input', required=True, help="Input file in JSON Lines format")
    parser.add_argument('-c', '--collection', required=True, help="Name of the MongoDB collection")
    args = parser.parse_args()

    # Initialize importer with command line arguments
    importer = JsonLinesImporter(args.input, mongo_uri=args.uri, db=database_name, collection=args.collection)
    importer.save_to_mongodb()
