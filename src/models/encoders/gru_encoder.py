import torch
import torch.nn as nn
from src.utils.config import Config

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=Config.GRU_HIDDEN_DIM, num_layers=Config.GRU_NUM_LAYERS, dropout=Config.GRU_DROPOUT):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pool = nn.AvgPool1d(4, 4)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (B, T, F)
        x = torch.tanh(self.proj(x))
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        return x
