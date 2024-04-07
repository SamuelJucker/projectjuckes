from flask import Flask, render_template
import matplotlib.pyplot as plt
import io
import base64
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model (adjust path as necessary)
model_path = 'C:/Users/jucke/Desktop/Juckesam/projectjuckes/model/predict/model_file2.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Dummy data for demonstration (replace with your actual data)
X_test = pd.DataFrame({'Sentiment': [0.1, 0.2, 0.3, 0.4, 0.5]})
y_test = pd.Series([0.15, 0.25, 0.35, 0.45, 0.55])

@app.route('/')
def plot():
    predictions = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Volatility')
    plt.plot(predictions, label='Predicted Volatility')
    plt.xlabel('Sample Index')
    plt.ylabel('Volatility')
    plt.title('Actual vs Predicted Volatility')
    plt.legend()

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Pass the encoded image to the template
    return render_template('plot.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
