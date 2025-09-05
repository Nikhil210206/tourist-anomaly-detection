import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance (in meters) between two lat/lon points."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of Earth in meters
    return c * r

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw GPS DataFrame and adds engineered features:
    - time differences
    - distances (via haversine)
    - speeds
    """
    # Sort by timestamp
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 1. Time difference between points
    df['time_diff_seconds'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

    # 2. Distance traveled between consecutive points
    df['prev_lat'] = df['latitude'].shift(1)
    df['prev_lon'] = df['longitude'].shift(1)
    df['distance_meters'] = df.apply(
        lambda row: haversine(row['prev_lat'], row['prev_lon'], row['latitude'], row['longitude'])
        if pd.notnull(row['prev_lat']) else 0,
        axis=1
    )

    # 3. Speed (m/s)
    df['speed_mps'] = df['distance_meters'] / df['time_diff_seconds']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df[['timestamp', 'latitude', 'longitude', 'time_diff_seconds', 'distance_meters', 'speed_mps']]

def prepare_training_data(df: pd.DataFrame, seq_len: int = 30):
    """
    Convert engineered features into sequences for LSTM training.

    Parameters:
    - df: DataFrame with GPS + features
    - seq_len: number of timesteps per sequence

    Returns:
    - X: numpy array of shape (samples, seq_len, features)
    - y: numpy array of shape (samples,) with anomaly labels
    """
    df = engineer_features(df)

    feature_cols = ['time_diff_seconds', 'distance_meters', 'speed_mps']
    data = df[feature_cols].values

    X, y = [], []

    # Sliding window
    for i in range(len(data) - seq_len):
        seq = data[i:i+seq_len]
        X.append(seq)

        # Basic labeling rule (can be replaced with real labels):
        avg_speed = np.mean(seq[:, 2])  # column 2 = speed
        if avg_speed < 0.1:  # near zero speed → inactivity anomaly
            y.append(1)  # anomaly
        else:
            y.append(0)  # normal

    return np.array(X), np.array(y)
