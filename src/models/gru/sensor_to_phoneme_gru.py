import torch
import torch.nn as nn
from src.models.decoders.attention_decoder import AttentionDecoder
from src.models.encoders.gru_encoder import GRUEncoder
from src.models.decoders.ctc_head import CTCHead
from src.utils.config import Config

class SensorToPhonemeGRU(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=Config.GRU_HIDDEN_DIM, num_layers=Config.GRU_NUM_LAYERS, dropout=Config.GRU_DROPOUT):
        super().__init__()
        self.encoder = GRUEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = CTCHead(hidden_dim, num_classes)
        # Attention decoder: encoder_dim matches hidden_dim
        self.attn_decoder = AttentionDecoder(encoder_dim=hidden_dim, hidden_dim=hidden_dim, num_classes=num_classes, sos_id=Config.SOS_ID, eos_id=Config.EOS_ID, pad_id=Config.PAD_ID)

    def forward(self, x, x_lengths=None, targets=None):
        # x: (B, T, F)
        enc_out = self.encoder(x)
        # enc_out: (B, T//4, H)
        
        # CTC Branch
        ctc_out = self.decoder(enc_out)
        ctc_logits_tbc = ctc_out.transpose(0, 1) # (T//4, B, C) for CTC
        
        # Attention Branch
        # Build encoder mask
        if x_lengths is not None:
            device = x.device
            max_t = enc_out.size(1)
            # x_lengths is original length. Encoder reduces by 4.
            encoder_mask = (torch.arange(max_t, device=device).unsqueeze(0) < (x_lengths // 4).unsqueeze(1))
        else:
            encoder_mask = torch.ones(enc_out.size(0), enc_out.size(1), dtype=torch.bool, device=enc_out.device)
            
        if not Config.USE_ATTENTION_DECODER:
            decoder_logits = None
        else:
            decoder_logits = self.attn_decoder(enc_out, encoder_mask, targets=targets)
        
        return ctc_logits_tbc, decoder_logits
