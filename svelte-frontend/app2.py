from flask import Flask, request, render_template
import os
from azure.storage.blob import BlobServiceClient
import pickle
import pandas as pd
import argparse
# from app2 import app, load_model
# from service import load_model_from_azure
# from your_module import load_model_from_azure


azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")


app = Flask(__name__)

# Global variable to store the model
model = None
model_file_path = load_model_from_azure(azure_storage_connection_string)
if model_file_path:
    with open(model_file_path, 'rb') as fid:
        model = pickle.load(fid)
else:
    print("Model could not be loaded from Azure.")
    # Handle the case where the model is not loaded appropriately


# Adjusted function to accept connection_string as an argument
def download_model(connection_string, container_name, blob_name, download_file_path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

# Adjusted function to accept connection_string as an argument
def load_model(connection_string):
    container_name = "juckesamblobvola"
    blob_name = "model_senti.pkl"
    download_file_path = "temp_model_senti.pkl"
    
    download_model(connection_string, container_name, blob_name, download_file_path)
    
    global model
    with open(download_file_path, "rb") as file:
        model = pickle.load(file)
    os.remove(download_file_path)  # Clean up the downloaded file

@app.route('/')
def home():
    return render_template('index.html')  # Assuming you have an index.html template


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded properly. Please check the connection string."

    # Assuming form data keys match the model's feature names
    feature_values = request.form.to_dict(flat=True)
    features_df = pd.DataFrame([feature_values], columns=model.feature_names_in_)  # Use model's expected feature names
    
    # Attempt prediction
    try:
        prediction = model.predict(features_df)
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return render_template('error.html', error_message=str(e))  # Consider creating an error template

    # Render a template with prediction result
    # return render_template('result.html', prediction=prediction[0])
