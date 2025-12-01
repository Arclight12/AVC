import torch
from torch.utils.data import Dataset
import numpy as np
import os

class BiosignalDataset(Dataset):
    def __init__(self, samples, tokenizer, preprocessor=None, original_rate=1000, augment=False):
        """
        Generic Biosignal Dataset.
        
        Args:
            samples (list): List of tuples (signal_path, label_path) or (signal_data, label_seq).
            tokenizer (PhonemeTokenizer): Tokenizer for labels.
            preprocessor (UniversalPreprocessor): Preprocessor for signals.
            original_rate (int): Sampling rate of input data.
            augment (bool): Whether to apply augmentation.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.original_rate = original_rate
        self.augment = augment
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        signal_source, label_source = self.samples[idx]
        
        # 1. Load Signal
        if isinstance(signal_source, str):
            # Handle file paths
            if signal_source.endswith('.adc'):
                # Assuming int16 binary for .adc (EMG-UKA style)
                # We need to know channels. For now, try to infer or assume 7 for .adc
                # This part is tricky to make fully generic without metadata.
                # We'll assume the user provides a loader or we infer from file extension + size.
                raw_data = np.fromfile(signal_source, dtype=np.int16)
                # Heuristic: try to reshape to (T, 7) or (T, 6)
                # For EMG-UKA, we know it's 7.
                # For generic, we might need metadata.
                # Let's assume 7 for .adc for now as per current project context.
                if raw_data.size % 7 == 0:
                    signal = raw_data.reshape(-1, 7)
                elif raw_data.size % 6 == 0:
                    signal = raw_data.reshape(-1, 6)
                else:
                    # Fallback or error
                    signal = raw_data.reshape(-1, 1) 
            elif signal_source.endswith('.npy'):
                signal = np.load(signal_source)
            else:
                raise ValueError(f"Unsupported file format: {signal_source}")
        else:
            # In-memory data
            signal = signal_source
            
        # 2. Preprocess
        if self.preprocessor:
            signal = self.preprocessor.process(signal, self.original_rate)
            
        # 3. Augment
        if self.augment:
            noise = np.random.normal(0, 0.05, signal.shape)
            signal = signal + noise
            
        # 4. Load Label
        if isinstance(label_source, str):
            # Load from file
            labels = []
            with open(label_source, 'r') as f:
                # Try to parse generic text or EMG-UKA format
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3: # EMG-UKA format: start end ph
                        ph = parts[2]
                        labels.append(ph)
                    elif len(parts) == 1: # Simple list
                        labels.append(parts[0])
        else:
            labels = label_source
            
        # 5. Encode
        label_ids = self.tokenizer.encode(labels)
        
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label_ids, dtype=torch.long)

def collate_fn_biosignal(batch):
    # batch is list of (signal_tensor, label_tensor)
    signals = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Pad signals
    lengths = torch.tensor([s.size(0) for s in signals], dtype=torch.long)
    max_len = max(lengths)
    input_dim = signals[0].size(1)
    
    padded_signals = torch.zeros(len(signals), max_len, input_dim)
    for i, s in enumerate(signals):
        padded_signals[i, :s.size(0), :] = s
        
    # Concatenate labels for CTC
    targets = torch.cat(labels)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    
    # Padded targets for Transformer
    max_target_len = max(target_lengths)
    padded_targets = torch.full((len(signals), max_target_len), 0, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_targets[i, :len(l)] = l
        
    return padded_signals, targets, lengths, target_lengths, padded_targets
