import argparse
import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceExistsError

# Argument parsing
parser = argparse.ArgumentParser(description='Upload Model to Azure Blob Storage')
parser.add_argument('-c', '--connection', required=True, help="Azure Storage connection string")
args = parser.parse_args()

try:
    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(args.connection)

    # Determine the base container name
    base_container_name = "juckesamblobvola"

    # Create or get the container
    try:
        blob_service_client.create_container(base_container_name)
        print(f"Container '{base_container_name}' created.")
    except ResourceExistsError:
        print(f"Container '{base_container_name}' already exists. Using existing.")

    # Set the base blob name
    base_blob_name = "model_senti.pkl"
    upload_file_path = os.path.join(".", base_blob_name)

    # Check if the blob exists and generate a new name if necessary
    container_client = blob_service_client.get_container_client(base_container_name)
    blobs_list = list(container_client.list_blobs())
    blob_names = [blob.name for blob in blobs_list]
    if base_blob_name in blob_names:
        # If the blob exists, append a suffix to create a new version
        suffix = 1
        new_blob_name = f"{base_blob_name.split('.')[0]}_{suffix}.pkl"
        while new_blob_name in blob_names:
            suffix += 1
            new_blob_name = f"{base_blob_name.split('.')[0]}_{suffix}.pkl"
        print(f"Blob '{base_blob_name}' already exists. Uploading as '{new_blob_name}' instead.")
        base_blob_name = new_blob_name
    else:
        print(f"Uploading blob as '{base_blob_name}'.")

    # Upload the blob
    blob_client = blob_service_client.get_blob_client(container=base_container_name, blob=base_blob_name)
    with open(file=upload_file_path, mode="rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"File '{base_blob_name}' uploaded successfully.")

except Exception as ex:
    print('Exception:', ex)
