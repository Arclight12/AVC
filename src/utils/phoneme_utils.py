import json
import os

class PhonemeTokenizer:
    def __init__(self, phoneme_list_file=None, phoneme_list=None):
        """
        Universal Phoneme Tokenizer.
        
        Args:
            phoneme_list_file (str): Path to JSON file containing list of phonemes.
            phoneme_list (list): List of phonemes.
        """
        self.phonemes = []
        if phoneme_list_file and os.path.exists(phoneme_list_file):
            with open(phoneme_list_file, 'r') as f:
                self.phonemes = json.load(f)
        elif phoneme_list:
            self.phonemes = phoneme_list
            
        # Ensure special tokens exist
        self.special_tokens = ["<BLANK>", "<SOS>", "<EOS>"]
        for t in self.special_tokens:
            if t not in self.phonemes:
                self.phonemes.append(t)
                
        self.ph2idx = {p: i for i, p in enumerate(self.phonemes)}
        self.idx2ph = {i: p for i, p in enumerate(self.phonemes)}
        self.blank_id = self.ph2idx["<BLANK>"]
        self.pad_id = self.blank_id # Alias for consistency
        self.sos_id = self.ph2idx["<SOS>"]
        self.eos_id = self.ph2idx["<EOS>"]
        
    def encode(self, phoneme_seq):
        """Encode a list of phonemes to indices."""
        return [self.ph2idx[p] for p in phoneme_seq if p in self.ph2idx]
        
    def decode(self, indices):
        """Decode a list of indices to phonemes."""
        return [self.idx2ph[int(i)] for i in indices if int(i) in self.idx2ph]
        
    def save(self, path):
        """Save phoneme list to JSON."""
        with open(path, 'w') as f:
            json.dump(self.phonemes, f)
            
    def __len__(self):
        return len(self.phonemes)
