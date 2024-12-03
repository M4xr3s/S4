import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_normalize_audio(train_folder, val_folder):
    def preprocess_audio_csv_files(folder_path):
        file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        all_audio_data = []
        all_valence = []

        for file_name in file_names:
            audio_file_path = os.path.join(folder_path, file_name)

            audio_data = pd.read_csv(audio_file_path, header=None, delimiter=';', usecols=range(2, 104))
            labels_data = pd.read_csv(audio_file_path, delimiter=';', usecols=['Valence'])

            all_audio_data.append(audio_data.values)
            all_valence.extend(labels_data['Valence'].tolist())

        all_audio_data = np.vstack(all_audio_data)
        all_valence = np.array(all_valence)

        return all_audio_data, all_valence

    def process_audio_folder(folder_path, scaler=None, fit_scaler=False):
        all_features, all_targets = [], []

        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                data = pd.read_csv(file_path)
                
                features = data.iloc[:, :-1].values
                targets = data.iloc[:, -1].values

                if fit_scaler:
                    scaler.fit(features)
                features = scaler.transform(features)
                
                all_features.append(features)
                all_targets.append(targets)

        features_array = np.vstack(all_features)
        targets_array = np.hstack(all_targets)
        return features_array, targets_array

    global_scaler = StandardScaler()

    audio_train, valence_train = preprocess_audio_csv_files(train_folder)
    global_scaler.fit(audio_train)

    features_train, targets_train = process_audio_folder(train_folder, scaler=global_scaler, fit_scaler=False)
    features_val, targets_val = process_audio_folder(val_folder, scaler=global_scaler, fit_scaler=False)

    return (features_train, targets_train), (features_val, targets_val)
