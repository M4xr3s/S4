import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

import os
import argparse


from s4 import S4Block
from tqdm.auto import tqdm

def ccc_loss(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    var_x = torch.var(x, unbiased=False)
    var_y = torch.var(y, unbiased=False)
    cov = torch.mean((x - mean_x) * (y - mean_y))
    ccc = (2 * cov) / (var_x + var_y + (mean_x - mean_y)**2)
    return 1 - ccc  # Return the loss


if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.1, type=float, help='Weight decay')
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=100, type=float, help='Training epochs')
# Dataset
parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'], type=str, help='Dataset')
parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
parser.add_argument('--d_model', default=256, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.2, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=1,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4Block(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        x = x.mean(dim=1)

        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

# Model
print('==> Building model..')
model = S4Model(
    d_input=2048,
    d_output=1,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=False,
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

criterion = ccc_loss
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)

import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader



import torchaudio
def speech_file_to_segments_fn(path, segment_length=3, hop_length=0.04):
    
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    speech_array = resampler(speech_array).squeeze()

    
    num_samples_per_segment = segment_length * 16000
    hop_samples = int(hop_length * 16000)

    
    segments = []
    for start in range(0, len(speech_array) - num_samples_per_segment + 1, hop_samples):
        end = start + num_samples_per_segment
        segment = speech_array[start:end]
        segments.append(segment.numpy())

    return segments

import pandas as pd

import numpy as np



def preprocess_function(wav_folder_path, csv_folder_path):
    
    wav_files = [os.path.join(wav_folder_path, f) for f in os.listdir(wav_folder_path) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} WAV files.")
    
    all_segments = []
    all_arousal = []
    all_valence = []

    for wav_file in wav_files:
        segments = speech_file_to_segments_fn(wav_file)
        base_name = os.path.basename(wav_file)
        csv_file_name = base_name.replace('.wav', '.csv')
        csv_file_path = os.path.join(csv_folder_path, csv_file_name)

        if os.path.exists(csv_file_path):
            print(f"Processing {wav_file} and corresponding CSV {csv_file_path}...")
            emotion_data = pd.read_csv(csv_file_path, delimiter=';', skiprows=1, names=['file_name', 'Time', 'Arousal', 'Valence'])
            emotion_data['Time'] = pd.to_numeric(emotion_data['Time'], errors='coerce')  

            for i, segment in enumerate(segments):
                time_stamp = i * 0.04  #
                print(f"Searching for rows between {time_stamp} and {time_stamp + 3} seconds.")
                matched_rows = emotion_data[(emotion_data['Time'] >= time_stamp) & (emotion_data['Time'] < time_stamp + 3)]
                if not matched_rows.empty:
                    row = matched_rows.iloc[0]
                    all_segments.append(segment)
                    all_arousal.append(row['Arousal'])
                    all_valence.append(row['Valence'])
                else:
                    print(f"No matching row for segment starting at {time_stamp} seconds.")
                    if i < 5:  
                        print(emotion_data.head())  #

            
        else:
             print(f"No corresponding CSV file found for {wav_file}.")
#
    return np.array(all_segments), np.array(all_arousal), np.array(all_valence)

input_wav = 'C:\\Users\\mahyl\\Documents\\recordings_audio\\recordings_audio'
input_csv = 'C:\\Users\\mahyl\\Documents\\Recola\\labels'
def preprocess_all_files(wav_folder_path, csv_folder_path):
    # List all WAV files in the folder
    wav_files = [os.path.join(wav_folder_path, f) for f in os.listdir(wav_folder_path) if f.endswith('.wav')]
    
    
    all_segments = []
    all_arousal = []
    all_valence = []

    for wav_file in wav_files:
        segments = speech_file_to_segments_fn(wav_file)

        
        base_name = os.path.basename(wav_file)
        csv_file_name = base_name.replace('.wav', '.csv')  
        csv_file_path = os.path.join(csv_folder_path, csv_file_name)

        
        emotion_data = pd.read_csv(csv_file_path, delimiter=';')
        emotion_data['Time'] = pd.to_numeric(emotion_data['Time'], errors='coerce')  

        for i, segment in enumerate(segments):
            time_stamp = i * 0.04  
            matched_rows = emotion_data[(emotion_data['Time'] >= time_stamp) & (emotion_data['Time'] < time_stamp + 3)]
            if not matched_rows.empty:
                row = matched_rows.iloc[0]
                all_segments.append(segment)
                all_arousal.append(row['Arousal'])
                all_valence.append(row['Valence'])

    return np.array(all_segments), np.array(all_arousal), np.array(all_valence)


wav_folder_path = input_wav
csv_folder_path = input_csv
segments, arousals, valences = preprocess_all_files(wav_folder_path, csv_folder_path)
segments = segments.reshape((segments.shape[0], 1, segments.shape[1]))
labels = np.hstack((arousals.reshape(-1,1), valences.reshape(-1,1)))


import os
import pandas as pd
import numpy as np

def preprocess_all_csv_files(audio_folder_path, ecg_folder_path, labels_folder_path):
    # List all files in the audio folder assuming that names are the same across folders
    file_names = [f for f in os.listdir(audio_folder_path) if f.endswith('.csv')]

    all_data = []
    all_arousal = []
    all_valence = []

    for file_name in file_names:
        audio_file_path = os.path.join(audio_folder_path, file_name)
        ecg_file_path = os.path.join(ecg_folder_path, file_name)
        labels_file_path = os.path.join(labels_folder_path, file_name)

        # Read the data from CSV files, skipping the first two columns
        audio_data = pd.read_csv(audio_file_path, header=None, delimiter=';', usecols=range(2, 104))
        ecg_data = pd.read_csv(ecg_file_path, header=None, delimiter=';', usecols=range(2, 56))
        labels_data = pd.read_csv(labels_file_path, delimiter=';', usecols=['Valence'])

        # Concatenate audio and ECG data along the columns
        combined_data = pd.concat([audio_data, ecg_data], axis=1)
        all_data.append(combined_data.values)  # Collect data across all files
        #all_arousal.extend(labels_data['Arousal'].tolist())
        all_valence.extend(labels_data['Valence'].tolist())

    # Convert list of arrays to a single array, flattening along the first dimension
    all_data = np.vstack(all_data)
    return all_data, np.array(all_valence),#np.array(all_arousal)



import os
import pandas as pd
import numpy as np

def preprocess_all_audio_files(audio_folder_path, labels_folder_path):
    # List all files in the audio folder assuming that names are the same across folders
    file_names = [f for f in os.listdir(audio_folder_path) if f.endswith('.csv')]

    all_audio_data = []
    all_valence = []

    for file_name in file_names:
        audio_file_path = os.path.join(audio_folder_path, file_name)
        labels_file_path = os.path.join(labels_folder_path, file_name)

        # Read the audio data from CSV files, skipping the first two columns
        audio_data = pd.read_csv(audio_file_path, header=None, delimiter=';', usecols=range(2, 56))
        labels_data = pd.read_csv(labels_file_path, delimiter=';', usecols=['Valence'])

        # Collect the audio data across all files
        all_audio_data.append(audio_data.values)
        all_valence.extend(labels_data['Valence'].tolist())

    # Convert list of arrays to a single array, flattening along the first dimension
    all_audio_data = np.vstack(all_audio_data)
    
    return all_audio_data, np.array(all_valence)

input_folder_pathh = 'C:\\Users\\mahyl\\Documents\\Recola\\features_ecg'
output_folder_pathh = 'C:\\Users\\mahyl\\Documents\\Recola\\labels'
audio_features, valeenciaga = preprocess_all_audio_files(input_folder_pathh, output_folder_pathh)
# Example usage:
#input_folder_path = 'C:\\Users\\mahyl\\Documents\\Recola\\features_audio'
#output_folder_path = 'C:\\Users\\mahyl\\Documents\\Recola\\features_ecg'
#data, arousal, valence = preprocess_all_csv_files(input_folder_path, output_folder_path)
import csv
import numpy as np
import os

def load_and_compile_data(folder_path):
    all_inputs = []
    all_outputs = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            inputs, outputs = load_data_excluding_last_rows(file_path, 2)
            all_inputs.extend(inputs)
            all_outputs.extend(outputs)
            print(f"Processed {filename}: Features shape {np.array(inputs).shape}, Output shape {np.array(outputs).shape}")

    # Convert lists to numpy arrays
    all_inputs_array = np.array(all_inputs)
    all_outputs_array = np.array(all_outputs)
    
    return all_inputs_array, all_outputs_array

def load_data_excluding_last_rows(file_path, rows_to_exclude):
    inputs = []
    outputs = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  
        
       
        rows = list(reader)
        for row in rows[:-rows_to_exclude]:  
            if row:  #
                inputs.append([float(x) for x in row[:-1]])
                outputs.append(float(row[-1]))

    return inputs, outputs

# Usage
folder_path = 'C:\\Users\\mahyl\\Documents\\Recola\\result_valence'
all_features, all_outputs = load_and_compile_data(folder_path)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = all_features  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.99)  
X_pca = pca.fit_transform(X_scaled)



input_folder_path = 'C:\\Users\\mahyl\\Documents\\Recola\\features_audio'
ecg_folder_path = 'C:\\Users\\mahyl\\Documents\\Recola\\features_ecg'
output_folder_path = 'C:\\Users\\mahyl\\Documents\\Recola\\labels'
input , valence =preprocess_all_csv_files(input_folder_path, ecg_folder_path, output_folder_path)
##########################################################################################
all_features = all_features.reshape((all_features.shape[0], 1, all_features.shape[1]))
all_outputs = all_outputs.reshape(-1,1)

class Wav(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]
   

batch_size = 128
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(all_features, all_outputs, test_size=0.2, random_state=42)

train_dataset = Wav(X_train_scaled, y_train)
test_dataset = Wav(X_test_scaled, y_test)

train_loader2 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader2 = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_losses = []
eval_losses = []

predictions = []
ground_truths = []
# Training
def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(train_loader2), total=len(train_loader2))
    for batch_idx, (inputs, targets) in pbar:
        inputs = inputs.to(torch.float32)
        targets = targets.to(torch.float32)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward() 
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f ' %
            (batch_idx, len(train_loader2), train_loss/(batch_idx+1))
        )
    average_train_loss = train_loss / len(train_loader2)
    train_losses.append(average_train_loss)

import pandas as pd

def eval(epoch, dataloader, checkpoint=True):
    global best_loss
    best_loss = float('inf')
    model.eval()
    eval_loss = 0
    total = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs = inputs.float()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            eval_loss += loss.item()
            total += targets.size(0)
            # Collecting predictions and targets
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f |' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1))
            )
    average_eval_loss = eval_loss / len(dataloader)
    eval_losses.append(average_eval_loss)      

    
    if checkpoint and eval_loss < best_loss:
        state = {
            'model': model.state_dict(),
            'loss': eval_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/s4-ckpt-256.pth')
        best_loss = eval_loss

    return average_eval_loss

    # Save checkpoint.
    if checkpoint and eval_loss < best_loss:
        state = {
            'model': model.state_dict(),
            'loss': eval_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/s4-ckpt-256.pth')
        best_loss = eval_loss

    results_df = pd.DataFrame({
    'Predicted Arousal': np.array(all_predictions).flatten(),
    'GT Arousal': np.array(all_targets).flatten()
})

    results_df.to_csv(f'predictions_epoch_{epoch}.csv', index=False)

    return eval_loss


pbar = tqdm(range(start_epoch, args.epochs))
if __name__ == "__main__":
    
    for epoch in pbar:
        pbar.set_description('Epoch: %d' % (epoch))
        train()
        eval(epoch, test_loader2)
        scheduler.step()
        # print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(eval_losses, label='Eval Loss')
    plt.title('Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()  

     # Plot predictions vs ground truth
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predicted')
    plt.plot(ground_truths, label='Ground Truth')
    plt.title('Predictions vs Ground Truth')
    plt.xlabel('Time')
    plt.ylabel('Valence')
    plt.legend()
    plt.show() 

