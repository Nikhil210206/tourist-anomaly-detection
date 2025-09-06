# scripts/train_lstm.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from app.model import create_lstm_model
from app.preprocessing import prepare_training_data

# --- Constants ---
RAW_DATA_PATH = "data/raw/sample_data.csv" # Use your real dataset path here
PROCESSED_X_PATH = "data/processed/X.npy"
PROCESSED_Y_PATH = "data/processed/y.npy"
MODEL_PATH = "models/lstm_multi_class_model.h5"

def load_or_create_data():
    """Loads processed data if it exists, otherwise creates it from raw data."""
    if os.path.exists(PROCESSED_X_PATH) and os.path.exists(PROCESSED_Y_PATH):
        print("Loading processed data from disk...")
        X = np.load(PROCESSED_X_PATH)
        y = np.load(PROCESSED_Y_PATH)
    else:
        print("Creating data from raw file...")
        df = pd.read_csv(RAW_DATA_PATH, parse_dates=['timestamp'])
        # You would add your anomaly simulation logic on the df here
        X, y = prepare_training_data(df)
        
        os.makedirs("data/processed", exist_ok=True)
        np.save(PROCESSED_X_PATH, X)
        np.save(PROCESSED_Y_PATH, y)
    return X, y

def main():
    X, y = load_or_create_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # Stratify helps with imbalanced classes
    )

    input_shape = X_train.shape[1:]
    model = create_lstm_model(input_shape=input_shape, num_classes=4)

    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss")

    print("--- Starting Model Training ---")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[checkpoint],
        verbose=1
    )
    print(f"✅ Training complete. Best model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()