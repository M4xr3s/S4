import os
import csv
import numpy as np

def load_data(file_path, start_exclusion=0):
    inputs, outputs = [], []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  
        for i, row in enumerate(reader):
            if i < start_exclusion:
                inputs.append([float(x) for x in row[:-1]])
                outputs.append(float(row[-1]))
    return np.array(inputs), np.array(outputs)

def load_folder_data(folder_path, start_exclusion=0):
    all_inputs, all_outputs = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            inputs, outputs = load_data(file_path, start_exclusion)
            all_inputs.append(inputs)
            all_outputs.append(outputs)
    return np.concatenate(all_inputs), np.concatenate(all_outputs)
