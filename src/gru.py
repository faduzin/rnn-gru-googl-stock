import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import (
    layers, 
    models,
    metrics, 
    optimizers, 
    losses, 
    callbacks
)

import random
random.seed(42)

def create_dataset(df, look_back=10):
    dataX, dataY = [], []
    for i in range(df.shape[0]-look_back):
        dataX.append(df[i:(i+look_back), :5])
        dataY.append(df[i + look_back, 3])
    return dataX, dataY


def build_rnn_gru(input_shape, output_shape):
    model = keras.Sequential()
    model.add(layers.GRU(15, input_shape=input_shape, activation="tanh", recurrent_activation="sigmoid", return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.GRU(15, activation="tanh", recurrent_activation="sigmoid"))
    model.add(layers.Dense(output_shape))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model


def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.show()