# app/main.py

from fastapi import FastAPI
import pandas as pd
from typing import List
from pydantic import BaseModel

# Import the functions from your other files
from .preprocessing import engineer_features
from .predictor import detect_rule_based_anomalies

app = FastAPI(title="Tourist Anomaly Detection API")

# Define the data structures for request and response
class GPSPoint(BaseModel):
    latitude: float
    longitude: float
    timestamp: str

class AnomalyRequest(BaseModel):
    user_id: str
    trajectory: List[GPSPoint]

# Create your API endpoint
@app.post("/detect-anomalies")
def detect_anomalies(request: AnomalyRequest):
    # 1. Convert the incoming request data into a Pandas DataFrame
    data = [p.dict() for p in request.trajectory]
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 2. Call the feature engineering function
    df_features = engineer_features(df)
    
    # 3. Call the rule-based detection function
    df_anomalies = detect_rule_based_anomalies(df_features)
    
    # 4. Prepare and return the response
    # For now, let's just return the last point's anomaly status
    last_point_anomalies = df_anomalies.iloc[-1].to_dict()

    return {"user_id": request.user_id, "analysis": last_point_anomalies}