import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from pymongo import MongoClient


def to_document(base_dir, item):
    try:
        ticker = item.get("ticker")
        provider_publish_time_str = item.get("providerPublishTime")
        model_result_is_positive = item.get("modelResultIsPositive")

        # Validate data types
        if not all(isinstance(x, (str, int, float)) for x in [ticker, provider_publish_time_str, model_result_is_positive]):
            raise ValueError("Invalid data types in item")

        # Parse datetime string
        provider_publish_time = datetime.strptime(provider_publish_time_str, "%Y-%m-%d %H:%M:%S")

        # Create document
        doc = {
        "ticker": ticker,
        "articleDate": provider_publish_time.date().strftime("%Y-%m-%d"),  # Convert date to string
        "providerPublishTime": provider_publish_time,
        "modelResultPositive": float(model_result_is_positive)
        }
        return doc

    except Exception as e:
        print(f"Error processing item: {e}")
        return None


class JsonLinesImporter:
    def __init__(self, file, mongo_uri, batch_size=30, db="juckesamDB", collection="juckesamCollection"):
        self.file = file
        self.base_dir = Path(file).parent
        self.batch_size = batch_size
        self.client = MongoClient(mongo_uri)
        self.db = db
        self.collection = collection

    def read_lines(self):
        with open(self.file, "r", encoding="UTF-8") as f:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--uri", required=True, help="MongoDB URI with username/password")
    parser.add_argument("-i", "--input", required=True, help="Input file in JSON Lines format")
    parser.add_argument("-c", "--collection", required=True, help="Name of the MongoDB collection")
    parser.add_argument("-d", "--db", default="juckesamDB", help="Name of the MongoDB database (default: juckesamDB)")
    args = parser.parse_args()
    file_path = r"C:\Users\jucke\Desktop\Juckesam\projectjuckes\model\Semantic\datasemantic\processed_articles.jl"
    importer = JsonLinesImporter(file_path, mongo_uri=args.uri, db=args.db, collection=args.collection)
    importer.save_to_mongodb()

    # Retrieve and print all entries from the collection (optional)
    db = importer.client[args.db]
    collection = db[args.collection]
    for doc in collection.find():
        print(doc)