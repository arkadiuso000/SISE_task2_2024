import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt


def load_data():
    columns = ["input_X", "input_Y", "expected_X", "expected_Y"]
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
            file_path = os.path.join(static_folder_path, file)
            temp_data_frame = pd.read_csv(file_path, names=columns)
            training_data = pd.concat([training_data, temp_data_frame], ignore_index=True)

        for file in dynamic_files:
            file_path = os.path.join(dynamic_folder_path, file)
            temp_data_frame = pd.read_csv(file_path, names=columns)
            testing_data = pd.concat([testing_data, temp_data_frame], ignore_index=True)

    # anty_null procedure
    training_data = training_data.dropna()
    testing_data = testing_data.dropna()

    # save to csv (debug purpose)
    training_data.to_csv("training_data.csv", index=False)
    testing_data.to_csv("testing_data.csv", index=False)

    print("Data loaded and saved to CSV ✅")
    return training_data, testing_data


def create_model(hidden_layers, activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform',num_of_inputs_neurons=2, num_of_outputs_neurons=2):
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
    ol = Dense(num_of_outputs_neurons, kernel_initializer=weight_init_method, activation=activation_function_out)
    model.add(ol)
    print("Model created ✅")
    return model

def train_model(model, training_data, epochs=100, learning_rate=0.01, optimizer='adam', moementum=0.9):
    train_data = training_data[["input_X", "input_Y"]]
    test_data = training_data[["expected_X", "expected_Y"]]

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate, momentum=moementum)
    else:
        raise ValueError('Optimizer must be either "adam" or "sgd"')

    model.compile(optimizer=opt, loss="mean_squared_error", metrics=['mse'])

    history = model.fit(train_data, test_data, epochs=epochs, validation_split=0.0, verbose=0)
    print("\tModel trained ✅")
    return history

def test_model(model, test_data):
    input_values = test_data[["input_X", "input_Y"]]
    validation_values = test_data[["expected_X", "expected_Y"]]

    mse = model.evaluate(input_values, validation_values, verbose=1)

    print("MSE on test data: {}".format(mse[1]))
    return mse[1]

def plot(history):
    plt.plot(history.history['mse'], label='MSE na zbiorze uczącym')
    # plt.plot(history.history['val_mse'], label='MSE na zbiorze walidacyjnym')
    plt.xlabel('Epoki')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
