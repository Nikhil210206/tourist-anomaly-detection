# app/model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape, num_classes=4):
    """
    Build and compile a multi-class LSTM model for anomaly detection.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(num_classes, activation="softmax") # Softmax for multi-class
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy", # Best for multi-class with integer labels
        metrics=["accuracy"]
    )
    model.summary()
    return model