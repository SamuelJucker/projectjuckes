import argparse
from app2 import app, load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Flask App with Azure Storage Connection String')
    parser.add_argument('-c', '--connection', required=True, help="Azure Storage connection string")
    args = parser.parse_args()

    # Load the model using the provided connection string
    load_model(args.connection)

    # Run the Flask app
    app.run(debug=True)
