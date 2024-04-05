# connection_strings.py

import os

# Use environment variables to store sensitive information
mongo_connection_string = os.getenv('MONGO_CONNECTION_STRING', "mongodb+srv://juckesam:S4m1lu%2B2006@mongojucke.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000")
database_name = os.getenv('DATABASE_NAME', 'juckesamDB')
collection_name = os.getenv('COLLECTION_NAME', 'juckesamCollection')


