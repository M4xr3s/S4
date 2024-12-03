import torch
from torch import nn
from transformers import Wav2Vec2Model

class Wav2VecRegressionModel(nn.Module):
    def __init__(self, wav2vec_name="facebook/wav2vec2-large-xlsr-53", output_dim=1):
        super(Wav2VecRegressionModel, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_name)
        self.regression_head = nn.Sequential(
            nn.Linear(self.wav2vec.config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        
        hidden_states = self.wav2vec(x).last_hidden_state
       
        pooled = hidden_states.mean(dim=1)
        
        return self.regression_head(pooled)
