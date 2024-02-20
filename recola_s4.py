'''
Train an S4 model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.
This code borrows heavily from https://github.com/kuangliu/pytorch-cifar.

This file only depends on the standalone S4 layer
available in /models/s4/

* Train standard sequential CIFAR:
    python -m example
* Train sequential CIFAR grayscale:
    python -m example --grayscale
* Train MNIST:
    python -m example --dataset mnist --d_model 256 --weight_decay 0.0

The `S4Model` class defined in this file provides a simple backbone to train S4 models.
This backbone is a good starting point for many problems, although some tasks (especially generation)
may require using other backbones.

The default CIFAR10 model trained by this file should get
89+% accuracy on the CIFAR10 test set in 80 epochs.

Each epoch takes approximately 7m20s on a T4 GPU (will be much faster on V100 / A100).
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import librosa
import torchvision
import io
import torchvision.transforms as transforms

import os
import argparse

from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
from tqdm.auto import tqdm
import torchaudio


# Dropout broke in PyTorch 1.11
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
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
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
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
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
        d_output=20,
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
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
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

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

# Model
print('==> Building model..')
model = S4Model(
    d_input=6,
    d_output=2,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
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


def ccc_loss(y_true, y_pred):
    true_mean = torch.mean(y_true)
    pred_mean = torch.mean(y_pred)
    covariance = torch.mean((y_pred - pred_mean) * (y_true - true_mean))
    true_var = torch.var(y_true)
    pred_var = torch.var(y_pred)
    ccc = (2. * covariance) / (pred_var + true_var + (pred_mean - true_mean) ** 2)
    return 1 - ccc


#criterion = nn.CrossEntropyLoss()
criterion = ccc_loss
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)


"""code for loading whugait dataset#"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
pd.set_option('display.max_rows', None)


class WhuGaitData(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]


audio_paths = ['./recordings_audio/train_1.wav', './recordings_audio/train_2.wav', './recordings_audio/train_3.wav', './recordings_audio/train_4.wav', './recordings_audio/train_5.wav', './recordings_audio/train_6.wav', './recordings_audio/train_7.wav', './recordings_audio/train_9.wav']
csv_paths = ['./labels/train_1.csv', './labels/train_2.csv', './labels/train_3.csv', './labels/train_4.csv', './labels/train_5.csv', './labels/train_6.csv', './labels/train_7.csv', './labels/train_8.csv', './labels/train_9.csv']

dev_audio_paths = ['./recordings_audio/dev_1.wav', './recordings_audio/dev_2.wav', './recordings_audio/dev_3.wav', './recordings_audio/dev_4.wav', './recordings_audio/dev_5.wav', './recordings_audio/dev_6.wav', './recordings_audio/dev_7.wav', './recordings_audio/dev_9.wav']
dev_csv_path = ['./labels/train_1.csv', './labels/train_2.csv', './labels/train_3.csv', './labels/train_4.csv', './labels/train_5.csv', './labels/train_6.csv', './labels/train_7.csv', './labels/train_8.csv', './labels/train_9.csv']

class AudioDataset(Dataset):
    def __init__(self, audio_paths = audio_paths, csv_paths = csv_paths, window_size=3, sr=16000, label_rate=0.04):
        self.audio_paths = audio_paths
        self.csv_paths = csv_paths
        self.window_size = window_size
        self.sr = sr
        self.label_rate = label_rate

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        waveform, _ = torchaudio.load(self.audio_paths[idx])
        labels = pd.read_csv(self.csv_paths[idx], skiprows=1, delimiter=';').iloc[:, 1:]
        labels_tensor = torch.tensor(labels.values, dtype=torch.float32)
        audio_segments, label_segments = self.process_audio_labels(waveform, labels_tensor)
        return audio_segments, label_segments

    def process_audio_labels(self, waveform, labels):
      
        waveform = self.normalize_waveform(waveform)    


        segment_length = self.window_size * self.sr
        hop_length = int(0.04 * self.sr)

        
        audio_segments = waveform.unfold(dimension=-1, size=segment_length, step=hop_length)
        
        
        labels_per_segment = int(self.window_size / self.label_rate)
        label_segments = [labels[i:i + labels_per_segment] for i in range(0, len(labels), labels_per_segment)]

        return audio_segments, label_segments
    
    def normalize_waveform(self, waveform):
        
        mean = waveform.mean()
        waveform -= mean

        std_dev = waveform.std()
        desired_variance = 0.5
        waveform = waveform / std_dev * torch.sqrt(torch.tensor(desired_variance))

        return waveform
    


train_loader = DataLoader(AudioDataset(audio_paths, csv_paths), batch_size=32, shuffle=True)
dev_loader = DataLoader(AudioDataset(dev_audio_paths, dev_csv_path), batch_size=32)

"""
# Create DataLoader for train and test datasets
batch_size = 128
test_dataset = WhuGaitData(X_test, y_test)
train_dataset = WhuGaitData(X_train, y_train)
train_loader2 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader2 = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
"""

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (inputs, targets) in pbar:
        inputs = inputs.float()
        print(inputs.shape())
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (batch_idx, len(train_loader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        )


def eval(epoch, dataloader, checkpoint=True):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs = inputs.float()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total)
            )

    # Save checkpoint.
    if checkpoint:
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_recola.pth')
            best_acc = acc

        return acc

pbar = tqdm(range(start_epoch, args.epochs))

if __name__ == "__main__":
    
    for epoch in pbar:
        pbar.set_description('Epoch: %d' % (epoch))
        train()
        eval(epoch, dev_loader)
        scheduler.step()
        # print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

