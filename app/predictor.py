import pandas as pd

def detect_rule_based_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a feature-engineered DataFrame and adds anomaly flags based on rules."""

    # Rule 1: Teleport
    df['anomaly_teleport'] = df['speed_mps'] > 100

    # Rule 2: Missing Updates
    df['anomaly_missing'] = df['time_diff_seconds'] > 600

    # Rule 3: Inactivity
    # We must set the index to use the time-based rolling window
    df.set_index('timestamp', inplace=True)
    df['distance_in_5min'] = df.rolling(window='300s')['distance_meters'].sum()
    df.reset_index(inplace=True) # Set it back to a column
    df['anomaly_inactive'] = df['distance_in_5min'] < 20
    
    return df