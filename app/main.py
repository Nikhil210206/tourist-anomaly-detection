# app/main.py

from fastapi import FastAPI
import pandas as pd
from typing import List
from pydantic import BaseModel
import numpy as np

# Import your functions and classes
from .preprocessing import engineer_features, prepare_training_data
from .predictor import AnomalyPredictor

app = FastAPI(title="Tourist Anomaly Detection API")

# --- Model Loading ---
# Load the model once when the API starts up.
# This is much more efficient than loading it on every request.
predictor = AnomalyPredictor()

# Define the data structures for request and response
class GPSPoint(BaseModel):
    latitude: float
    longitude: float
    timestamp: str

class AnomalyRequest(BaseModel):
    user_id: str
    trajectory: List[GPSPoint]

@app.post("/detect-anomalies")
def detect_anomalies(request: AnomalyRequest):
    # 1. Convert incoming data to a DataFrame
    data = [p.dict() for p in request.trajectory]
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 2. Engineer features
    df_features = engineer_features(df)
    
    # 3. Prepare the last part of the trajectory as a sequence for the model
    # The model expects a sequence of a specific length (e.g., 30)
    SEQ_LEN = 30 
    feature_cols = ['time_diff_seconds', 'distance_meters', 'speed_mps']
    
    # Ensure there are enough data points
    if len(df_features) < SEQ_LEN:
        return {"error": f"Not enough data points. Need {SEQ_LEN}, got {len(df_features)}"}

    # Get the last sequence
    last_sequence = df_features[feature_cols].tail(SEQ_LEN).values
    
    # 4. Get the ML model's prediction
    ml_prediction = predictor.predict(last_sequence)

    return {
        "user_id": request.user_id,
        "ml_analysis": ml_prediction
    }