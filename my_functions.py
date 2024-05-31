import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
