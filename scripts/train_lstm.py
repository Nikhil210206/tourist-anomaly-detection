import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from app.model import create_lstm_model
from app import preprocessing  # <-- reuse your preprocessing functions

DATA_X = "data/processed/X.npy"
DATA_Y = "data/processed/y.npy"
MODEL_PATH = "models/lstm_model.h5"

def load_data():
    """
    Load processed features and labels.
    Uses preprocessing.py if raw data exists.
    """
    if os.path.exists(DATA_X) and os.path.exists(DATA_Y):
        X = np.load(DATA_X)
        y = np.load(DATA_Y)
    else:
        # You can implement data preparation inside preprocessing.py
        X, y = preprocessing.prepare_training_data()
        os.makedirs("data/processed", exist_ok=True)
        np.save(DATA_X, X)
        np.save(DATA_Y, y)

    return X, y

def main():
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_shape = X_train.shape[1:]  # (timesteps, features)
    model = create_lstm_model(input_shape=input_shape, num_classes=2)

    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss")

    history = model.fit(
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
