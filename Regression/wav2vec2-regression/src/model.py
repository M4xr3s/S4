import torch
from torch import nn

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pooling(x).squeeze(-1)
        x = self.activation(self.fc1(x))
        return self.fc2(x)

def ccc_loss(x, y):
    mean_x, mean_y = torch.mean(x), torch.mean(y)
    var_x, var_y = torch.var(x), torch.var(y)
    cov = torch.mean((x - mean_x) * (y - mean_y))
    ccc = (2 * cov) / (var_x + var_y + (mean_x - mean_y) ** 2)
    return 1 - ccc
