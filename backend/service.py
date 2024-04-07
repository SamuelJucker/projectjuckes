import os
import pickle
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from azure.storage.blob import BlobServiceClient
import datetime

# Load model function
def load_model_from_azure(azure_storage_connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
    container_name = None

    # Identify the latest container
    print("Fetching blob containers...")
    containers = blob_service_client.list_containers(include_metadata=True)
    latest_suffix = 0
    for container in containers:
        if container['name'].startswith("juckesamblobv"):
            parts = container['name'].split("-")
            if len(parts) == 3 and parts[-1].isdigit() and int(parts[-1]) > latest_suffix:
                latest_suffix = int(parts[-1])
                container_name = container['name']
    # container_name = "juckesamblobvola"
    # blob_name = "model_senti.pkl"
    # download_file_path = "temp_model_senti.pkl"
    
    if container_name:
        container_client = blob_service_client.get_container_client(container_name)
        print(f"Using container: {container_name}")
        
        # Assume single blob with model in the container
        blob_list = list(container_client.list_blobs())
        if blob_list:
            blob_name = blob_list[0].name
            download_file_path = os.path.join("../model", blob_name)
            print(f"Downloading blob to: {download_file_path}")

            os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
            with open(download_file_path, "wb") as download_file:
                download_file.write(container_client.download_blob(blob_name).readall())
                
            return download_file_path
    else:
        print("No suitable container found.")
    return None

# Init app
app = Flask(__name__)
CORS(app)
app = Flask(__name__, static_url_path='', static_folder='../frontend/build')

# Load the model
azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
model_file_path = load_model_from_azure(azure_storage_connection_string) if azure_storage_connection_string else "../model/GradientBoostingRegressor.pkl"
with open(model_file_path, 'rb') as fid:
    model = pickle.load(fid)

@app.route("/")
def indexPage():
    return send_file("../frontend/build/index.html")

@app.route("/api/predict")
def predict_route():
    downhill = request.args.get('downhill', default=0, type=int)
    uphill = request.args.get('uphill', default=0, type=int)
    length = request.args.get('length', default=0, type=int)

    demo_input = [[downhill, uphill, length, 0]]
    demo_df = pd.DataFrame(columns=['downhill', 'uphill', 'length_3d', 'max_elevation'], data=demo_input)
    demo_output = model.predict(demo_df)
    time = demo_output[0]

    return jsonify({
        'time': str(datetime.timedelta(seconds=time)),
        'din33466': str(datetime.timedelta(seconds=din33466(uphill=uphill, downhill=downhill, distance=length))),
        'sac': str(datetime.timedelta(seconds=sac(uphill=uphill, downhill=downhill, distance=length)))
    })

if __name__ == "__main__":
    app.run(debug=True)
