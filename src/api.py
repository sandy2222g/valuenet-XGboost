from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Create the FastAPI app
app = FastAPI()

# Define the input data model using Pydantic
class HouseFeatures(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: float
    nox: float
    rm: float
    age: float
    dis: float
    rad: float
    tax: float
    ptratio: float
    b: float
    lstat: float

# Load the trained model
model_path = os.path.join("models", "xgb_model.joblib")
model = joblib.load(model_path)

@app.get("/")
def read_root():
    """
    Root endpoint for health check.
    """
    return {"message": "House Price Prediction API is running."}

@app.post("/predict/")
def predict_price(features: HouseFeatures):
    """
    Prediction endpoint.
    Receives house features and returns the predicted price.
    """
    # Convert input data to a pandas DataFrame
    data = pd.DataFrame([features.dict()])
    
    # Make prediction
    prediction = model.predict(data)
    
    return {"predicted_price": prediction[0]}

