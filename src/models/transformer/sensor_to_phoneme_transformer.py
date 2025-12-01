import torch
import torch.nn as nn
from src.models.encoders.transformer_encoder import TransformerEncoder, PositionalEncoding
from src.utils.config import Config

class SensorToPhonemeTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=Config.TRANSFORMER_D_MODEL, nhead=Config.TRANSFORMER_NHEAD, num_encoder_layers=Config.TRANSFORMER_NUM_ENCODER_LAYERS, num_decoder_layers=Config.TRANSFORMER_NUM_DECODER_LAYERS, dim_feedforward=Config.TRANSFORMER_DIM_FEEDFORWARD, dropout=Config.TRANSFORMER_DROPOUT):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        
        # Decoder
        self.tgt_emb = nn.Embedding(num_classes + 2, d_model) 
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.out = nn.Linear(d_model, num_classes)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # src: (B, T_src, F)
        # tgt: (B, T_tgt)
        
        memory = self.encoder(src, src_key_padding_mask) # (T_src, B, d_model)
        
        tgt = self.tgt_emb(tgt) # (B, T, d_model)
        tgt = tgt.permute(1, 0, 2) # (T, B, d_model)
        tgt = self.pos_decoder(tgt)
        
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, 
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=src_key_padding_mask)
        
        output = self.out(output) # (T, B, C)
        return output.permute(1, 0, 2) # (B, T, C)
