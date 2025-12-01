import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.config import Config
from src.utils.ctc_decoder import ctc_best_path_decode
from src.utils.metrics import compute_per, compute_accuracy
from src.losses.hybrid_loss import hybrid_loss, lambda_schedule

class Trainer:
    def __init__(self, model, train_loader, val_loader, tokenizer, config, device, dataset_type="unknown"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.dataset_type = dataset_type
        
        self.model_type = "transformer"
        if "GRU" in model.__class__.__name__:
             self.model_type = "gru"
             self.criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
             self.ce_criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_ID)
        else:
             self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
                 
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        # Calculate lambda for this epoch
        total_epochs = self.config.EPOCHS
        lambda_ctc = lambda_schedule(epoch, total_epochs, start=self.config.LAMBDA_START, mid=self.config.LAMBDA_MID, end=self.config.LAMBDA_END)
        
        # Failsafe for synthetic data
        use_hybrid = self.config.USE_HYBRID_LOSS and self.config.USE_ATTENTION_DECODER
        if self.dataset_type == "synthetic":
            use_hybrid = False
        
        for batch_idx, (padded_signals, targets, lengths, target_lengths, padded_targets) in enumerate(self.train_loader):
            # Enforce GPU
            padded_signals = padded_signals.to(self.config.DEVICE)
            targets = targets.to(self.config.DEVICE)
            lengths = lengths.to(self.config.DEVICE)
            target_lengths = target_lengths.to(self.config.DEVICE)
            padded_targets = padded_targets.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            
            if self.model_type == "gru":
                lengths_reduced = (lengths // 4).clamp(min=1).to(self.config.DEVICE)
                
                if use_hybrid:
                    ctc_logits, decoder_logits = self.model(padded_signals, x_lengths=lengths, targets=padded_targets)
                else:
                    ctc_logits, _ = self.model(padded_signals, x_lengths=lengths, targets=None)
                    decoder_logits = None
                
                # CTC Loss
                ctc_loss = self.criterion(ctc_logits, targets, lengths_reduced, target_lengths)
                
                if use_hybrid and decoder_logits is not None:
                    # Attention Loss
                    B, L, C = decoder_logits.size()
                    attn_loss = self.ce_criterion(decoder_logits.view(B*L, C), padded_targets.view(-1))
                    loss = hybrid_loss(ctc_loss, attn_loss, lambda_ctc=lambda_ctc)
                else:
                    loss = ctc_loss
                
            else: # Transformer
                # ... (Transformer logic same as before) ...
                # Re-construct targets for Transformer
                B = padded_targets.shape[0]
                targets_list = []
                for i in range(B):
                    t = padded_targets[i]
                    t = t[t != -1]
                    targets_list.append(t)
                    
                max_len_batch = max([len(t) for t in targets_list]) + 1
                dec_input_batch = torch.full((B, max_len_batch), 0, dtype=torch.long).to(self.config.DEVICE)
                target_batch = torch.full((B, max_len_batch), -1, dtype=torch.long).to(self.config.DEVICE)
                
                for i, t in enumerate(targets_list):
                    l = len(t)
                    dec_input_batch[i, 0] = self.tokenizer.sos_id
                    dec_input_batch[i, 1:l+1] = t.to(self.config.DEVICE)
                    target_batch[i, :l] = t.to(self.config.DEVICE)
                    
                src_key_padding_mask = (torch.arange(padded_signals.size(1))[None, :] >= lengths[:, None]).to(self.config.DEVICE)
                
                tgt_key_padding_mask = torch.zeros((B, max_len_batch), dtype=torch.bool).to(self.config.DEVICE)
                for i, t in enumerate(targets_list):
                    tgt_key_padding_mask[i, len(t)+1:] = True
                    
                output = self.model(padded_signals, dec_input_batch, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
                loss = self.criterion(output.reshape(-1, len(self.tokenizer)), target_batch.reshape(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self):
        if not self.val_loader:
            return
            
        self.model.eval()
        total_per = 0
        total_acc = 0
        count = 0
        
        with torch.no_grad():
            for batch_idx, (padded_signals, targets, lengths, target_lengths, padded_targets) in enumerate(self.val_loader):
                padded_signals = padded_signals.to(self.device)
                lengths = lengths.to(self.device)
                
                if self.model_type == "gru":
                    # Forward
                    # Note: validate doesn't pass targets usually, but for attention decoder inference we don't need them
                    # However, the model signature is forward(x, x_lengths, targets=None)
                    # If targets is None, it does greedy inference.
                    ctc_logits, decoder_logits = self.model(padded_signals, x_lengths=lengths, targets=None)
                    
                    # Prefer Attention Decoder if available
                    if decoder_logits is not None:
                        # decoder_logits: (B, L_pred, C)
                        attn_preds = decoder_logits.argmax(dim=-1).cpu().tolist()
                        
                        pred_ids = []
                        for seq in attn_preds:
                            # Trim at EOS
                            clean_seq = []
                            for token in seq:
                                if token == self.tokenizer.eos_id:
                                    break
                                if token != self.tokenizer.pad_id and token != self.tokenizer.sos_id:
                                    clean_seq.append(token)
                            pred_ids.append(clean_seq)
                    else:
                        # Fallback to CTC
                        # ctc_logits: (T, B, C) or (B, T, C) depending on model output
                        # Model returns (T//4, B, C) for CTC
                        # Let's transpose to (B, T//4, C) for decoder if needed, or check ctc_best_path_decode
                        # ctc_best_path_decode expects (B, T, C) usually? Let's check.
                        # In previous code: logp_batch = logp.transpose(0, 1) -> (B, T, C)
                        ctc_logits_batch = ctc_logits.transpose(0, 1)
                        pred_ids = ctc_best_path_decode(ctc_logits_batch, blank_id=self.tokenizer.blank_id)
                    
                    # Ground Truth
                    # targets is flat tensor. target_lengths tells us how to slice.
                    true_ids = []
                    start = 0
                    for l in target_lengths:
                        t = targets[start:start+l].tolist()
                        true_ids.append(t)
                        start += l
                        
                    # Metrics
                    per = compute_per(pred_ids, true_ids)
                    acc = compute_accuracy(per)
                    
                    total_per += per
                    total_acc += acc
                    count += 1
                else:
                    # Transformer validation (skip for now or implement greedy)
                    pass

        if count > 0:
            avg_per = total_per / count
            avg_acc = total_acc / count
            print(f"  Validation PER: {avg_per:.4f} (Accuracy: {avg_acc:.2f}%)")
            return avg_per, avg_acc
        return 0, 0

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
