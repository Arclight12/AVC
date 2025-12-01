import torch
import numpy as np
from src.preprocessing.universal_preprocessing import UniversalPreprocessor
from src.models.gru.sensor_to_phoneme_gru import SensorToPhonemeGRU
from src.models.transformer.sensor_to_phoneme_transformer import SensorToPhonemeTransformer
from src.utils.config import Config
from src.utils.ctc_decoder import ctc_best_path_decode

class InferenceEngine:
    def __init__(self, model_type, checkpoint_path, tokenizer, device=None):
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.device = device if device else torch.device(Config.DEVICE)
        self.preprocessor = UniversalPreprocessor(target_rate=Config.TARGET_SAMPLING_RATE)
        self.model = None
        self.checkpoint_path = checkpoint_path

    def load_model(self, input_dim):
        num_classes = len(self.tokenizer)
        if self.model_type == "gru":
            # Using default config params for now, ideally saved in checkpoint
            self.model = SensorToPhonemeGRU(input_dim, num_classes).to(self.device)
        else:
            self.model = SensorToPhonemeTransformer(input_dim, num_classes).to(self.device)
            
        if self.checkpoint_path:
            try:
                self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
                print(f"Loaded checkpoint: {self.checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
        self.model.eval()

    def predict(self, signal):
        """
        Predict phonemes from a raw signal (numpy array).
        Signal shape: (Channels, Time) or (Time, Channels)
        """
        # Lazy Load
        if self.model is None:
             # Infer input dim from signal
             if signal.shape[0] < signal.shape[1] and signal.shape[0] < 20: # (C, T)
                 input_dim = signal.shape[0]
             else: # (T, C)
                 input_dim = signal.shape[1]
             self.load_model(input_dim)

        # Preprocess
        processed_signal = self.preprocessor.process(signal, original_rate=1000) # (T, C)
        
        # To Tensor
        input_tensor = torch.tensor(processed_signal, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, T, C)
        
        # Inference
        with torch.no_grad():
            if self.model_type == "gru":
                # Forward
                ctc_logits, decoder_logits = self.model(input_tensor)
                
                # Prefer Attention Decoder
                if decoder_logits is not None:
                    # decoder_logits: (1, L, C)
                    attn_preds = decoder_logits.argmax(dim=-1).cpu().tolist()[0]
                    
                    # Clean up attention preds
                    pred_ids = []
                    for token in attn_preds:
                        if token == self.tokenizer.eos_id:
                            break
                        if token != self.tokenizer.pad_id and token != self.tokenizer.sos_id:
                            pred_ids.append(token)
                else:
                    pred_ids = []
                        
                # Fallback to CTC if attention is empty or not available
                if len(pred_ids) == 0:
                     # CTC Decode
                     ctc_logits_batch = ctc_logits.transpose(0, 1) # (1, T, C)
                     pred_ids_batch = ctc_best_path_decode(ctc_logits_batch, blank_id=self.tokenizer.blank_id)
                     pred_ids = pred_ids_batch[0]
                
            else:
                return ["TRANSFORMER_NOT_IMPL"]
                
        # Decode to Phonemes
        phonemes = self.tokenizer.decode(pred_ids)
        return phonemes
