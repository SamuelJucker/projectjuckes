import os
import pickle
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from azure.storage.blob import BlobServiceClient
import datetime
import re


# Define a function to load the model from Azure Blob Storage
def load_model_from_azure(azure_storage_connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
    container_name = "juckesamblobvola"
    container_client = blob_service_client.get_container_client(container_name)

    print(f"Fetching blobs from container: {container_name}")
    blob_list = list(container_client.list_blobs())
    
    # Use a regular expression to find blobs that match the pattern 'model_senti_<number>.pkl'
    model_blobs = {}
    for blob in blob_list:
        match = re.match(r"model_senti_(\d+)\.pkl", blob.name)
        if match:
            model_blobs[int(match.group(1))] = blob.name
    
    # Find the blob with the highest number
    if model_blobs:
        latest_blob_name = model_blobs[max(model_blobs)]
        download_file_path = os.path.join("model", latest_blob_name)
        Path("model").mkdir(parents=True, exist_ok=True)
        print(f"Downloading the latest model blob to: {download_file_path}")

        with open(download_file_path, "wb") as download_file:
            download_file.write(container_client.download_blob(latest_blob_name).readall())
        
        return download_file_path
    else:
        print("No model blobs found.")
        return None


app = Flask(__name__, static_url_path='', static_folder='../svelte-frontend/public')
CORS(app)

# Load the model
azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
model_file_path = load_model_from_azure(azure_storage_connection_string) if azure_storage_connection_string else "model/GradientBoostingRegressor.pkl"
with open(model_file_path, 'rb') as fid:
    model = pickle.load(fid)

@app.route("/")
def indexPage():
    return send_file("../svelte-frontend/public/index.html")
# return send_file("../frontend/public/index.html")


@app.route("/api/predict", methods=['GET'])
def predict_route():
    # Retrieve query parameters
    open_price = request.args.get('open', default=0.0, type=float)
    bid = request.args.get('bid', default=0.0, type=float)
    ask = request.args.get('ask', default=0.0, type=float)
    marketCap = request.args.get('marketCap', default=0.0, type=float)
    beta = request.args.get('beta', default=0.0, type=float)
    trailingPE = request.args.get('trailingPE', default=0.0, type=float)
    volume = request.args.get('volume', default=0.0, type=float)
    modelResultPositive = request.args.get('modelResultPositive', default=0.5, type=float)

    # Prepare the features dataframe as expected by the model
    features = pd.DataFrame([{
        'open': open_price, 
        'bid': bid, 
        'ask': ask, 
        'marketCap': marketCap, 
        'beta': beta, 
        'trailingPE': trailingPE, 
        'volume': volume, 
        'modelResultPositive': modelResultPositive
    }])
    
    # Predict
    prediction = model.predict(features)

    # Return the prediction in JSON format
    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)


