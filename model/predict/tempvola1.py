import pandas as pd
import argparse
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
import plotly.express as px
import pickle

class StockDataRetriever:
    def __init__(self, mongo_uri, db_name, collection_name):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def fetch_data(self):
        cursor = self.collection.find({"modelResultPositive": {"$exists": False}})
        data = pd.DataFrame(list(cursor))
        return data

    def fetch_sentiment_data(self):
        cursor = self.collection.find({"modelResultPositive": {"$exists": True}})
        df = pd.DataFrame(list(cursor))
        if not df.empty:
            df['modelResultPositive'] = df.get('modelResultPositive', 0.5)
            sentiment_avg = df.groupby('ticker', as_index=False)['modelResultPositive'].mean()
            sentiment_avg['modelResultPositive'] = sentiment_avg['modelResultPositive'].fillna(0.5)
        else:
            sentiment_avg = pd.DataFrame(columns=['ticker', 'modelResultPositive'])
        return sentiment_avg

def preprocess_data(stock_data, sentiment_data):
    merged_df = pd.merge(stock_data, sentiment_data, on='ticker', how='left')
    merged_df['modelResultPositive'] = merged_df['modelResultPositive'].fillna(0.5)
    
    indicators = ['open', 'bid', 'ask', 'marketCap', 'beta', 'trailingPE', 'volume', 'modelResultPositive']
    for indicator in indicators[:-1]:  # Last indicator is sentiment, no conversion needed
        merged_df[indicator] = pd.to_numeric(merged_df[indicator].replace('N/A', np.nan), errors='coerce')
    
    merged_df.dropna(subset=indicators[:-1] + ['previousClose'], inplace=True)  # Exclude sentiment from dropping N/A
    
    features = merged_df[indicators]
    target = merged_df['previousClose'].astype(float)
    return features, target

def train_model(features, target):
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2, X_test, y_test, y_pred

def visualize_results(X_test, y_test, y_pred):
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predictions'})
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                             mode='lines', line=dict(color='red'), name='Ideal Fit'))
    fig.update_layout(title='Gradient Boosting Model Predictions vs Actual Values', xaxis_title='Actual', yaxis_title='Predicted')
    fig.show()

def save_model(model, filename="model_senti.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(model, file)

# After training the model in the 'if __name__ == "__main__":' block


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Data Analysis')
    parser.add_argument('-u', '--uri', required=True, help="MongoDB URI with username/password")
    args = parser.parse_args()

    retriever = StockDataRetriever(args.uri, "juckesamDB", "juckesamCollection")
    stock_data = retriever.fetch_data()
    sentiment_data = retriever.fetch_sentiment_data()

    if not stock_data.empty:
        features, target = preprocess_data(stock_data, sentiment_data)
        model, mse, r2, X_test, y_test, y_pred = train_model(features, target)
        print(f"Model MSE: {mse}")
        print(f"Model R^2: {r2}")
        
        visualize_results(X_test, y_test, y_pred)
    else:
        print("No data found.")


if not stock_data.empty:
    features, target = preprocess_data(stock_data, sentiment_data)
    model, mse, r2, X_test, y_test, y_pred = train_model(features, target)
    print(f"Model MSE: {mse}")
    print(f"Model R^2: {r2}")
    
    visualize_results(X_test, y_test, y_pred)
    
    # Save the model to a file
    save_model(model, "model_senti.pkl")
    print("Model saved to model_senti.pkl")
else:
    print("No data found.")