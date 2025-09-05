import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape, num_classes=2):
    """
    Build and compile an LSTM model for anomaly detection.

    Parameters:
    - input_shape: (timesteps, features)
    - num_classes: 2 (normal vs anomaly) or more if multiple anomaly types

    Returns:
    - Compiled LSTM model
    """
    model = Sequential()

    # LSTM stack
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))

    model.add(LSTM(32))
    model.add(Dropout(0.3))

    # Output
    if num_classes == 2:
        model.add(Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(Dense(num_classes, activation="softmax"))
        loss = "categorical_crossentropy"

    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=["accuracy"]
    )

    return model
