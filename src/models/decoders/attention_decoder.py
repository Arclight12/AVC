import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionDecoder(nn.Module):
    """
    Luong-style attention decoder. Compatible with encoder_outputs shape (B, T_enc, E).
    Returns logits: (B, L, num_classes) in training (targets provided) or (B, max_len, num_classes) in inference.
    """
    def __init__(self, encoder_dim, hidden_dim, num_classes, sos_id=1, eos_id=2, pad_id=0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.embedding = nn.Embedding(num_classes, hidden_dim, padding_idx=pad_id)
        self.attn_proj = nn.Linear(encoder_dim, hidden_dim, bias=False)
        self.gru = nn.GRU(hidden_dim + encoder_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, encoder_outputs, encoder_mask, targets=None, max_len=200, teacher_forcing_ratio=0.9):
        """
        encoder_outputs: (B, T_enc, E)
        encoder_mask: (B, T_enc) bool tensor, True for valid positions
        targets: (B, L) LongTensor or None
        returns: logits (B, L, num_classes) (training) or (B, max_len, num_classes) (inference)
        """
        B, T_enc, E = encoder_outputs.size()
        device = encoder_outputs.device

        # initial hidden state
        h = torch.zeros(1, B, self.hidden_dim, device=device)

        # preproject encoder for fast attention scoring
        proj_enc = self.attn_proj(encoder_outputs)  # (B, T_enc, hidden)

        def attend(h_t):
            # h_t: (B, hidden)
            # scores: (B, T_enc)
            scores = torch.bmm(proj_enc, h_t.unsqueeze(-1)).squeeze(-1)
            scores = scores.masked_fill(~encoder_mask, float('-inf'))
            attn_w = F.softmax(scores, dim=-1)  # (B, T_enc)
            context = torch.bmm(attn_w.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, E)
            return context

        logits = []
        if targets is not None:
            L = targets.size(1)
            prev_ids = torch.full((B,), self.sos_id, dtype=torch.long, device=device)
            for t in range(L):
                use_teacher = torch.rand(1).item() < teacher_forcing_ratio
                if t == 0:
                    input_ids = prev_ids
                else:
                    input_ids = targets[:, t-1] if use_teacher else pred_ids

                emb = self.embedding(input_ids)  # (B, hidden)
                context = attend(h[0])           # (B, E)
                rnn_in = torch.cat([emb, context], dim=-1).unsqueeze(1)  # (B,1,hidden+E)
                out, h = self.gru(rnn_in, h)     # out: (B,1,hidden)
                out = out.squeeze(1)
                logit = self.out(out)            # (B, num_classes)
                pred_ids = torch.argmax(logit, dim=-1)
                logits.append(logit.unsqueeze(1))
            return torch.cat(logits, dim=1)      # (B, L, num_classes)
        else:
            # inference (greedy)
            prev_ids = torch.full((B,), self.sos_id, dtype=torch.long, device=device)
            for _ in range(max_len):
                emb = self.embedding(prev_ids)
                context = attend(h[0])
                rnn_in = torch.cat([emb, context], dim=-1).unsqueeze(1)
                out, h = self.gru(rnn_in, h)
                out = out.squeeze(1)
                logit = self.out(out)
                prev_ids = torch.argmax(logit, dim=-1)
                logits.append(logit.unsqueeze(1))
            return torch.cat(logits, dim=1)
