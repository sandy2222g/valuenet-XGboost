import joblib
import pandas as pd
import os
from src.data_loader import load_data

def predict_price():
    """
    Loads the trained model and makes a prediction on sample data.
    """
    # Load model
    model_path = os.path.join("models", "xgb_model.joblib")
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first by running 'python main.py --action train'")
        return

    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # Get sample data (first row of the dataset)
    df = load_data()
    sample_data = df.drop(['price'], axis=1).iloc[[0]]
    
    # Make prediction
    prediction = model.predict(sample_data)
    print(f"Sample Data:\n{sample_data.to_string()}")
    print(f"\nPredicted price for the sample data: {prediction[0]}")

if __name__ == '__main__':
    predict_price()
