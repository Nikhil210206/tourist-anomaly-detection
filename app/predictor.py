# app/predictor.py
import os
import numpy as np
import tensorflow as tf
from app.model import create_lstm_model

MODEL_PATH = "models/lstm_multi_class_model.h5"
ANOMALY_LABELS = {
    0: "Normal",
    1: "Prolonged Inactivity",
    2: "Sudden Drop-off / Teleport",
    3: "Deviation from Planned Route"
}

class AnomalyPredictor:
    def __init__(self):
        """Initializes the predictor by loading the trained model."""
        if os.path.exists(MODEL_PATH):
            self.model = tf.keras.models.load_model(MODEL_PATH)
        else:
            # This is a fallback for development. In production, the model must exist.
            print(f"WARNING: Model file not found at {MODEL_PATH}. Using a dummy model.")
            self.model = None

    def predict(self, sequence: np.ndarray):
        """Predicts the anomaly type for a given sequence."""
        if self.model is None:
            return {"error": "Model not loaded"}

        # Reshape for a batch of 1
        sequence = sequence.reshape(1, *sequence.shape)
        prediction = self.model.predict(sequence, verbose=0)[0]

        anomaly_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "anomaly_detected": anomaly_class != 0,
            "anomaly_type": ANOMALY_LABELS.get(anomaly_class, "Unknown"),
            "confidence_score": confidence
        }