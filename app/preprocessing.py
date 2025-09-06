# app/preprocessing.py

import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    # ... (your existing haversine function is perfect, no changes needed) ...
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 # Radius of Earth in meters
    return c * r

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... (your existing engineer_features function is perfect, no changes needed) ...
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['time_diff_seconds'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    df['prev_lat'] = df['latitude'].shift(1)
    df['prev_lon'] = df['longitude'].shift(1)
    df['distance_meters'] = df.apply(
        lambda row: haversine(row['prev_lat'], row['prev_lon'], row['latitude'], row['longitude'])
        if pd.notnull(row['prev_lat']) else 0,
        axis=1
    )
    df['speed_mps'] = df['distance_meters'] / df['time_diff_seconds']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # NEW: Add a simulated planned route for deviation detection
    # This creates a simple straight line between the first and last point
    start_point = df.iloc[0]
    end_point = df.iloc[-1]
    df['planned_lat'] = np.linspace(start_point['latitude'], end_point['latitude'], len(df))
    df['planned_lon'] = np.linspace(start_point['longitude'], end_point['longitude'], len(df))
    df['deviation_meters'] = df.apply(
        lambda row: haversine(row['latitude'], row['longitude'], row['planned_lat'], row['planned_lon']),
        axis=1
    )
    
    return df[['timestamp', 'latitude', 'longitude', 'time_diff_seconds', 'distance_meters', 'speed_mps', 'deviation_meters']]


def prepare_training_data(df: pd.DataFrame, seq_len: int = 30):
    """
    Converts engineered features into sequences and generates MULTI-CLASS labels.
    Labels: 0=Normal, 1=Inactive, 2=Teleport, 3=Deviation
    """
    df_features = engineer_features(df)
    
    feature_cols = ['time_diff_seconds', 'distance_meters', 'speed_mps', 'deviation_meters']
    data = df_features[feature_cols].values

    X, y = [], []

    for i in range(len(data) - seq_len):
        seq_features = data[i:i + seq_len]
        
        # --- Labeling Logic ---
        # We check for anomalies in the LAST point of the sequence
        last_point_speed = seq_features[-1, 2] # speed is the 3rd feature
        last_point_deviation = seq_features[-1, 3] # deviation is the 4th feature
        
        # Check for anomalies in order of priority
        if last_point_speed > 100:  # Teleport/Sudden Drop-off
            label = 2
        elif last_point_deviation > 200: # Deviation from route (200 meters)
            label = 3
        elif last_point_speed < 0.1 and np.mean(seq_features[:, 2]) < 1.0: # Inactivity
            label = 1
        else: # If none of the above, it's normal
            label = 0
            
        X.append(seq_features)
        y.append(label)

    return np.array(X), np.array(y)