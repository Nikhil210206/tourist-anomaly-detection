import os
import numpy as np
import tensorflow as tf
from app.model import create_lstm_model

MODEL_PATH = "models/lstm_model.h5"

class AnomalyPredictor:
    def __init__(self, input_shape=(30, 4), num_classes=2):
        """
        Initialize the predictor. Loads a trained model if available.
        """
        if os.path.exists(MODEL_PATH):
            self.model = tf.keras.models.load_model(MODEL_PATH)
        else:
            # Create a fresh model (useful during development before training)
            self.model = create_lstm_model(input_shape=input_shape, num_classes=num_classes)

    def predict(self, sequence: np.ndarray):
        """
        Predict anomaly on a given sequence.

        Parameters:
        - sequence: numpy array of shape (timesteps, features)

        Returns:
        - dict with anomaly status, confidence score
        """
        sequence = sequence.reshape(1, *sequence.shape)  # reshape for batch
        prediction = self.model.predict(sequence, verbose=0)[0]

        if self.model.output_shape[-1] == 1:
            # Binary case
            anomaly_prob = float(prediction)
            return {
                "anomaly": anomaly_prob > 0.5,
                "confidence_score": anomaly_prob if anomaly_prob > 0.5 else 1 - anomaly_prob
            }
        else:
            # Multi-class case
            anomaly_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            return {
                "anomaly": anomaly_class != 0,
                "anomaly_type": anomaly_class,
                "confidence_score": confidence
            }
