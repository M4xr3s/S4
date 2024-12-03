import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_normalize_ecg_and_audio(train_folder, val_folder):
    def preprocess_csv_files(folder_path, feature_type):
        file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        all_data = []
        all_valence = []

        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)

            if feature_type == "audio":
                data = pd.read_csv(file_path, header=None, delimiter=';', usecols=range(2, 104))
            elif feature_type == "ecg":
                data = pd.read_csv(file_path, header=None, delimiter=';', usecols=range(2, 56))

            labels = pd.read_csv(file_path, delimiter=';', usecols=['Valence'])

            all_data.append(data.values)
            all_valence.extend(labels['Valence'].tolist())

        all_data = np.vstack(all_data)
        all_valence = np.array(all_valence)

        return all_data, all_valence

    def process_folder(folder_path, scaler, feature_type):
        all_features = []
        all_targets = []

        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                data = pd.read_csv(file_path)

                if feature_type == "audio":
                    features = data.iloc[:, :102].values  
                elif feature_type == "ecg":
                    features = data.iloc[:, :54].values  
                targets = data.iloc[:, -1].values  
                features = scaler.transform(features)

                all_features.append(features)
                all_targets.append(targets)

        features_array = np.vstack(all_features)
        targets_array = np.hstack(all_targets)
        return features_array, targets_array

    audio_scaler = StandardScaler()
    ecg_scaler = StandardScaler()

    audio_train, valence_train = preprocess_csv_files(train_folder, "audio")
    ecg_train, _ = preprocess_csv_files(train_folder, "ecg")

    audio_scaler.fit(audio_train)
    ecg_scaler.fit(ecg_train)

    audio_features_train, audio_targets_train = process_folder(train_folder, audio_scaler, "audio")
    ecg_features_train, _ = process_folder(train_folder, ecg_scaler, "ecg")

    audio_features_val, audio_targets_val = process_folder(val_folder, audio_scaler, "audio")
    ecg_features_val, _ = process_folder(val_folder, ecg_scaler, "ecg")

    return {
        "train": {"audio": (audio_features_train, audio_targets_train), "ecg": ecg_features_train},
        "val": {"audio": (audio_features_val, audio_targets_val), "ecg": ecg_features_val}
    }
