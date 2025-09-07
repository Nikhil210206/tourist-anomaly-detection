# app/preprocessing.py

import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 # Radius of Earth in meters
    return c * r

# This is the original function, still needed for the training data generation.
def prepare_training_data(df: pd.DataFrame, seq_len: int = 30):
    feature_cols = ['time_diff_seconds', 'distance_meters', 'speed_mps', 'deviation_meters']
    data = df[feature_cols].values

    X, y = [], []

    for i in range(len(data) - seq_len):
        seq_features = data[i:i + seq_len]
        
        last_point_speed = seq_features[-1, 2]
        last_point_deviation = seq_features[-1, 3]
        
        if last_point_speed > 100:
            label = 2
        elif last_point_deviation > 200:
            label = 3
        elif last_point_speed < 0.1 and np.mean(seq_features[:, 2]) < 1.0:
            label = 1
        else:
            label = 0
            
        X.append(seq_features)
        y.append(label)

    return np.array(X), np.array(y)

# --- NEW HIGH-ACCURACY FUNCTIONS ---

def find_closest_point(point, route_points):
    """Helper function to find the minimum distance from a point to a route."""
    distances = np.array([haversine(point['latitude'], point['longitude'], rp['latitude'], rp['longitude']) for rp in route_points])
    return np.min(distances)

def engineer_features_live(df_live: pd.DataFrame, df_planned: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for a live trajectory by comparing it to an actual planned route.
    """
    df_live = df_live.sort_values(by='timestamp').reset_index(drop=True)
    
    # Calculate basic features as before
    df_live['time_diff_seconds'] = df_live['timestamp'].diff().dt.total_seconds().fillna(0)
    df_live['prev_lat'] = df_live['latitude'].shift(1)
    df_live['prev_lon'] = df_live['longitude'].shift(1)
    df_live['distance_meters'] = df_live.apply(
        lambda row: haversine(row['prev_lat'], row['prev_lon'], row['latitude'], row['longitude'])
        if pd.notnull(row['prev_lat']) else 0,
        axis=1
    )
    df_live['speed_mps'] = df_live['distance_meters'] / df_live['time_diff_seconds']
    df_live.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_live.fillna(0, inplace=True)
    
    # --- The High-Accuracy Deviation Calculation ---
    planned_points = df_planned[['latitude', 'longitude']].to_dict('records')
    
    if not planned_points: # Handle empty planned route
        df_live['deviation_meters'] = 0.0
    else:
        df_live['deviation_meters'] = df_live.apply(
            lambda row: find_closest_point(row, planned_points),
            axis=1
        )
    
    return df_live[['timestamp', 'latitude', 'longitude', 'time_diff_seconds', 'distance_meters', 'speed_mps', 'deviation_meters']]