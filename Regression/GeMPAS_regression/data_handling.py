import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_normalize_csv(train_folder, val_folder):
   
  
    def process_folder(folder_path):
        all_features, all_targets = [], []
        scaler = StandardScaler()
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                data = pd.read_csv(file_path)
                
                features = data.iloc[:, :-1].values  
                targets = data.iloc[:, -1].values   
                
                features = scaler.fit_transform(features)
            
                all_features.append(features)
                all_targets.append(targets)
        
        
        features_array = np.vstack(all_features)
        targets_array = np.hstack(all_targets)
        return features_array, targets_array
    
    features_train, targets_train = process_folder(train_folder)
    features_val, targets_val = process_folder(val_folder)
    
    return (features_train, targets_train), (features_val, targets_val)
