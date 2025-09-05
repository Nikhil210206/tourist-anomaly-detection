import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 # Radius of earth in meters.
    return c * r

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a raw GPS DataFrame and adds features."""
    
    # Make sure data is sorted by timestamp before doing anything
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # 1. Calculate Time Difference
    df['time_diff_seconds'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # 2. Calculate Distance
    df['prev_lat'] = df['latitude'].shift(1)
    df['prev_lon'] = df['longitude'].shift(1)
    df['distance_meters'] = df.apply(
        lambda row: haversine(row['prev_lat'], row['prev_lon'], row['latitude'], row['longitude'])
        if pd.notnull(row['prev_lat'])
        else 0,
        axis=1
    )
    
    # 3. Calculate Speed
    df['speed_mps'] = df['distance_meters'] / df['time_diff_seconds']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Clean up and return
    return df[['timestamp', 'latitude', 'longitude', 'time_diff_seconds', 'distance_meters', 'speed_mps']]