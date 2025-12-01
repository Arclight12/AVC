import torch
import torch.nn as nn
import math
from src.utils.config import Config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=Config.TRANSFORMER_D_MODEL, nhead=Config.TRANSFORMER_NHEAD, num_layers=Config.TRANSFORMER_NUM_ENCODER_LAYERS, dim_feedforward=Config.TRANSFORMER_DIM_FEEDFORWARD, dropout=Config.TRANSFORMER_DROPOUT):
        super().__init__()
        self.src_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask=None):
        # src: (B, T, F)
        src = self.src_proj(src) # (B, T, d_model)
        src = src.permute(1, 0, 2) # (T, B, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output # (T, B, d_model)
