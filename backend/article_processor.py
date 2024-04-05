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


# app = Flask(__name__)

# # Setup MongoDB connection
# client = MongoClient(mongo_connection_string)
# db = client[database_name]
# collection = db[collection_name]

# # Initialize tokenizer and ONNX model
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# model_path = "C:/Users/jucke/Desktop/Juckesam/projectjuckes/model/Semantic/roberta-sequence-classification-9.onnx"
# ort_session = ort.InferenceSession(model_path)

# def sentiment_analysis(text):
#     """Function to perform sentiment analysis on the given text."""
#     input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
#     ort_inputs = {ort_session.get_inputs()[0].name: input_ids.detach().cpu().numpy()}
#     ort_outs = ort_session.run(None, ort_inputs)
#     pred = np.argmax(ort_outs)
#     return "positive" if pred == 1 else "negative"

# @app.route("/process", methods=["POST"])
# def process_articles():
#     """Endpoint to process articles from a .jl file and update MongoDB with sentiment analysis results."""
#     # For simplicity, the articles path is hardcoded; you might want to accept it via request payload
#     articles_path = "C:/Users/jucke/Desktop/Juckesam/projectjuckes/data/articles.jl"
#     with open(articles_path, 'r') as file:
#         for line in file:
#             article = json.loads(line)
#             text = article.get('articleText')
#             sentiment = sentiment_analysis(text)
#             # Update document with sentiment analysis result
#             collection.update_one({"_id": article['_id']}, {"$set": {"sentiment": sentiment}}, upsert=True)
#     return "Processing completed."

# if __name__ == "__main__":
#     app.run(debug=True)

# # Assuming your connection_strings.py is correctly named and accessible
# from connectionstrings import mongo_connection_string, database_name, collection_name

# app = Flask(__name__)

# # Setup MongoDB connection
# client = MongoClient(mongo_connection_string)
# db = client[database_name]
# collection = db[collection_name]

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# model_path = "C:/Users/jucke/Desktop/Juckesam/projectjuckes/model/Semantic/roberta-sequence-classification-9.onnx"
# ort_session = ort.InferenceSession(model_path)

# @app.route("/process", methods=["POST"])
# def process_articles():
#     articles_path = "C:/Users/jucke/Desktop/Juckesam/projectjuckes/data/articles.jl"
#     with open(articles_path, 'r') as file:
#         for line in file:
#             article = json.loads(line)
#             text = article.get('articleText')
#             input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
#             ort_inputs = {ort_session.get_inputs()[0].name: input_ids.detach().cpu().numpy()}
#             ort_outs = ort_session.run(None, ort_inputs)
#             pred = np.argmax(ort_outs)
#             sentiment = "positive" if pred == 1 else "negative"
#             # Update document with sentiment analysis
#             collection.update_one({"_id": article['_id']}, {"$set": {"sentiment": sentiment}})
#     return "Processing completed."

# if __name__ == "__main__":
#     app.run(debug=True)


# # # Import MongoDB connection details
# # from connectionstrings import mongo_connection_string, database_name, collection_name

# # app = Flask(__name__)

# # # Initialize tokenizer and ONNX model
# # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# # model_path = r"C:\Users\jucke\Desktop\Juckesam\projectjuckes\model\Semantic\roberta-sequence-classification-9.onnx"
# # ort_session = onnxruntime.InferenceSession(model_path)

# # # MongoDB setup using imported connection details
# # client = MongoClient(mongo_connection_string)
# # db = client[database_name]
# # collection = db[collection_name]

# # def sentiment_analysis(text):
# #     input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
# #     ort_inputs = {ort_session.get_inputs()[0].name: input_ids.numpy()}
# #     ort_outs = ort_session.run(None, ort_inputs)
# #     pred = np.argmax(ort_outs[0])
# #     return "positive" if pred == 1 else "negative"

# # @app.route("/process-articles")
# # def process_articles():
# #     with open(r'C:\Users\jucke\Desktop\Juckesam\projectjuckes\smi_scraper\data\articles.jl', 'r') as file:
# #         for line in file:
# #             article = json.loads(line)
# #             sentiment = sentiment_analysis(article['articleText'])
# #             article['sentiment'] = sentiment
# #             # Store in MongoDB
# #             collection.insert_one(article)
# #             # Write to a new .jl file
# #             with open(r'C:\Users\jucke\Desktop\Juckesam\projectjuckes\smi_scraper\data\categorized_articles.jl', 'a') as outfile:
# #                 json.dump(article, outfile)
# #                 outfile.write('\n')
# #     return "Articles processed and stored."

# # if __name__ == "__main__":
# #     app.run(debug=True)
