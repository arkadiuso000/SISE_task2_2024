import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt


def load_data():
    columns = ["intup_X", "intup_Y", "excpected_X", "excpected_Y"]
    training_data = pd.DataFrame(columns=columns)
    testing_data = pd.DataFrame(columns=columns)

    static_folders_paths = ["./dane/f8/stat", "./dane/f10/stat"]
    dynamic_folders_paths = ["./dane/f8/dyn", "./dane/f10/dyn"]

    for i in range(2):
        static_folder_path = static_folders_paths[i]
        dynamic_folder_path = dynamic_folders_paths[i]

        # Sorting the files to ensure data is concatenated in alphabetical order.
        static_files = sorted(file for file in os.listdir(static_folder_path) if file.endswith(".csv"))
        dynamic_files = sorted(file for file in os.listdir(dynamic_folder_path) if file.endswith(".csv"))

        for file in static_files:
            if file.endswith(".csv"):
                file_path = os.path.join(static_folder_path, file)
                temp_data_frame = pd.read_csv(file_path, names=columns)
                training_data = pd.concat([training_data, temp_data_frame], ignore_index=True)

        for file in dynamic_files:
            if file.endswith(".csv"):
                file_path = os.path.join(dynamic_folder_path, file)
                temp_data_frame = pd.read_csv(file_path, names=columns)
                testing_data = pd.concat([testing_data, temp_data_frame], ignore_index=True)

    return training_data, testing_data

def create_model(num_of_inputs_neurons, hidden_layers, num_of_outputs_neurons=2, activation_function='tanh', weight_init_method='glorot_uniform'):
    # sequential model allows creating model layer by layer
    model = Sequential()

    # input layer
    il = Input(shape=(num_of_inputs_neurons,))
    model.add(il)

    # first hidden layer
    fhl = Dense(hidden_layers[0], activation=activation_function, kernel_initializer=weight_init_method)
    model.add(fhl)

    for layer in hidden_layers[1:]:
        # next hidden layer
        nhl = Dense(layer, activation=activation_function, kernel_initializer=weight_init_method)
        model.add(nhl)

    # output layer
    ol = Dense(num_of_outputs_neurons, kernel_initializer=weight_init_method)
    model.add(ol)
    return model

def train_model(model, training_data, epochs=100, learning_rate=0.01, optimizer='adam'):
    train_data = training_data[["intup_X", "intup_Y"]]
    test_data = training_data[["excpected_X", "excpected_Y"]]

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    else:
        raise ValueError('Optimizer must be either "adam" or "sgd"')

    model.compile(optimizer=opt, loss="mean_squared_error", metrics=['mse'])

    history = model.fit(train_data, test_data, epochs=epochs, validation_split=0.2, verbose=0)
    return history