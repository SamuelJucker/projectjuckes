import sys
sys.path.append(r'C:\Users\jucke\Desktop\Juckesam\projectjuckes')

from flask import Flask
import json
import torch
import numpy as np
from transformers import RobertaTokenizer
from pymongo import MongoClient

import os
from pymongo import MongoClient
import onnxruntime as ort
from transformers import RobertaTokenizer
import numpy as np

# Assuming connection_strings.py is in the Python path or same directory
from connectionstrings import mongo_connection_string, database_name, collection_name
from flask import Flask, request, send_file
import torch

from pymongo import MongoClient


import requests

response = requests.post('http://127.0.0.1:5000/process')
print(response.text)

# Adjust the sys.path to include the directory where connectionstrings.py is located
sys.path.append(r'C:\Users\jucke\Desktop\Juckesam\projectjuckes')

# Import MongoDB connection details
from connectionstrings import mongo_connection_string, database_name, collection_name

app = Flask(__name__)

# Setup MongoDB connection
client = MongoClient(mongo_connection_string)
db = client[database_name]
collection = db[collection_name]

# Initialize tokenizer and ONNX model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model_path = "C:/Users/jucke/Desktop/Juckesam/projectjuckes/model/Semantic/roberta-sequence-classification-9.onnx"
ort_session = ort.InferenceSession(model_path)

def sentiment_analysis(text):
    """Function to perform sentiment analysis on the given text."""
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
    ort_inputs = {ort_session.get_inputs()[0].name: input_ids.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    pred = np.argmax(ort_outs)
    return "positive" if pred == 1 else "negative"

@app.route("/process", methods=["POST"])
def process_articles():
    """Endpoint to process articles from a .jl file and update MongoDB with sentiment analysis results."""
    articles_path = "C:/Users/jucke/Desktop/Juckesam/projectjuckes/data/articles.jl"
    processed_articles_path = "C:/Users/jucke/Desktop/Juckesam/projectjuckes/data/processed_articles.jl"  # New path for processed articles
    with open(articles_path, 'r') as file, open(processed_articles_path, 'w') as outfile:
        for line in file:
            article = json.loads(line)
            text = article.get('articleText')
            sentiment = sentiment_analysis(text)
            # Update document with sentiment analysis result in MongoDB
            collection.update_one({"_id": article['_id']}, {"$set": {"sentiment": sentiment}}, upsert=True)
            # Save updated article to a new .jl file
            article['sentiment'] = sentiment  # Add sentiment result to the article
            json.dump(article, outfile)
            outfile.write('\n')
    return "Processing and saving completed."

if __name__ == "__main__":
    app.run(debug=True)

