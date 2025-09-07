# app/main.py

from fastapi import FastAPI
import pandas as pd
from typing import List, Optional
from pantic import BaseModel
import numpy as np

# Import your functions and the predictor class
from .preprocessing import engineer_features_live
from .predictor import AnomalyPredictor

app = FastAPI(title="Tourist Anomaly Detection API")

# --- Model Loading ---
# Load the model once when the API starts up for efficiency.
predictor = AnomalyPredictor()

# --- Data Structures ---
# Defines the format for a single GPS point.
class GPSPoint(BaseModel):
    latitude: float
    longitude: float
    timestamp: Optional[str] = None # Timestamp is optional for the planned route points

# Defines the format for the incoming request from the main application.
class AnomalyRequest(BaseModel):
    user_id: str
    live_trajectory: List[GPSPoint]
    planned_trajectory: List[GPSPoint] # The actual planned route from your backend

# --- API Endpoint ---
@app.post("/detect-anomalies")
def detect_anomalies(request: AnomalyRequest):
    # 1. Convert the incoming JSON data into Pandas DataFrames
    live_data = [p.dict() for p in request.live_trajectory]
    planned_data = [p.dict() for p in request.planned_trajectory]

    df_live = pd.DataFrame(live_data)
    df_live['timestamp'] = pd.to_datetime(df_live['timestamp'])
    
    df_planned = pd.DataFrame(planned_data)

    # 2. Call the new, high-accuracy feature engineering function
    df_features = engineer_features_live(df_live, df_planned)
    
    # 3. Prepare the last 30 points as a sequence for the model
    SEQ_LEN = 30
    feature_cols = ['time_diff_seconds', 'distance_meters', 'speed_mps', 'deviation_meters']
    
    if len(df_features) < SEQ_LEN:
        return {"error": f"Not enough data points. Need {SEQ_LEN}, got {len(df_features)}"}

    last_sequence = df_features[feature_cols].tail(SEQ_LEN).values
    
    # 4. Get the prediction from the loaded ML model
    ml_prediction = predictor.predict(last_sequence)

    return {
        "user_id": request.user_id,
        "ml_analysis": ml_prediction
    }