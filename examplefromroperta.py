import json
import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime as ort
from pymongo import MongoClient
from connectionstrings import mongo_connection_string, database_name, collection_name

# Initialize tokenizer and ONNX model session
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model_path = r"C:\Users\jucke\Desktop\Juckesam\sentiment\roberta-sequence-classification-9.onnx"
ort_session = ort.InferenceSession(model_path)

# Function to predict sentiment of a given text
def predict_sentiment(text):
    # Truncate text to the model's max input size (512 tokens for RoBERTa)
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
    input_ids = torch.tensor([input_ids])  # Add batch dimension
    ort_inputs = {ort_session.get_inputs()[0].name: input_ids.cpu().numpy()}
    try:
        ort_outs = ort_session.run(None, ort_inputs)
        pred = np.argmax(ort_outs)
        # Return 1 for positive sentiment, 0 for negative
        return 1 if pred == 1 else 0
    except Exception as e:
        print(f"Error during sentiment prediction: {e}")
        return "error"  # Use a different value to indicate an error

def process_articles(input_path, output_path, max_articles=1100):
    articles_processed = 0
    with open(input_path, 'r', encoding='utf-8') as input_file, open(output_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            if articles_processed >= max_articles:
                break
            article = json.loads(line)
            text = article.get("articleText", "")
            sentiment = predict_sentiment(text)
            if sentiment != "error":
                processed_article = {
                    "ticker": article.get("ticker", ""),
                    "providerPublishTime": article.get("providerPublishTime", ""),
                    "articleText": text[:300],  # First 20 characters of article text
                    "modelResultIsPositive": sentiment
                }
                json.dump(processed_article, output_file)
                output_file.write('\n')
                articles_processed += 1
    print(f"Processed and stored {articles_processed} articles.")

if __name__ == "__main__":
    input_path = r"C:\Users\jucke\Desktop\Juckesam\projectjuckes\smi_scraper\data\articles.jl"
    output_path = r"C:\Users\jucke\Desktop\Juckesam\projectjuckes\model\Semantic\datasemantic\processed_articles.jl"
    process_articles(input_path, output_path)
