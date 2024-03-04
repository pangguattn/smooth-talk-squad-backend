import numpy as np
import time
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Normalization, Input, Dense, LSTM, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential()
    model.add(Input(shape=(20, 47)))
    model.add(Normalization(axis=-1))
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(LSTM(units=6, activation="tanh", input_shape=(20, 47)))
    model.add(Dense(units=12, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = Adam(lr=0.001)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall(), AUC()],
    )

    print("✅ Model compiled")

    return model


def train_model(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size=256,
    patience=2,
    validation_data=None,  # overrides validation_split
    validation_split=0.3,
) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(patience=15, restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001
    )

    history = model.fit(
        X, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es, reduce_lr]
    )

    print(
        f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}"
    )

    return model, history


def evaluate_model(
    model: Model, X: np.ndarray, y: np.ndarray, batch_size=64
) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True,
    )

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
