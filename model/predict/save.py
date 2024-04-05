import argparse
import os, uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Argument parsing
connection_string = "mongodb+srv://jucke:S4m1lu%2B2006@mongojucke.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
parser = argparse.ArgumentParser(description='Upload Model to Azure Blob Storage')
parser.add_argument('-c', '--connection', required=True, help="Azure Storage connection string")
args = parser.parse_args()

try:
    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(args.connection)

    # Determine a unique container name (for example purposes, we're using a fixed name)
    container_name = "juckesamblob"

    # Create the container if it does not exist
    blob_service_client.create_container(container_name)

    local_file_name = "model_file.pkl"
    upload_file_path = os.path.join(".", local_file_name)

    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)
    print(f"\nUploading to Azure Storage as blob:\n\t{local_file_name}")

    # Upload the created file
    with open(file=upload_file_path, mode="rb") as data:
        blob_client.upload_blob(data)

    print(f"File '{local_file_name}' uploaded successfully.")

except Exception as ex:
    print('Exception:', ex)
