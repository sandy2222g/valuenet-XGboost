import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import joblib
import os

from src.data_loader import load_data

def train_model():
    """
    Trains the XGBoost model and saves it.
    """
    # Load data
    df = load_data()

    # Split data
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # Train model
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Evaluate model
    test_prediction = model.predict(X_test)
    r2 = metrics.r2_score(y_test, test_prediction)
    mae = metrics.mean_absolute_error(y_test, test_prediction)
    print(f"R-squared error on test set: {r2}")
    print(f"Mean Absolute Error on test set: {mae}")

    # Save model
    model_path = os.path.join("models", "xgb_model.joblib")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()
