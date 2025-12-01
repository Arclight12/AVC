import argparse
import torch
import numpy as np
from src.inference.infer_core import InferenceEngine
from src.utils.phoneme_utils import PhonemeTokenizer
from src.utils import EMG_UKA_PHONEMES

def run_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    phoneme_list = list(EMG_UKA_PHONEMES.keys())
    tokenizer = PhonemeTokenizer(phoneme_list=phoneme_list)
    
    engine = InferenceEngine(args.model_type, args.checkpoint, tokenizer, device)
    
    # Mock Input
    if args.input_file:
        if args.input_file.endswith('.adc'):
            raw_data = np.fromfile(args.input_file, dtype=np.int16)
            if raw_data.size % 7 == 0:
                signal = raw_data.reshape(-1, 7)
            elif raw_data.size % 6 == 0:
                signal = raw_data.reshape(-1, 6)
            else:
                # Fallback
                signal = raw_data.reshape(-1, 1)
        elif args.input_file.endswith('.npy'):
            signal = np.load(args.input_file)
        else:
            raise ValueError("Unsupported file format")
    else:
        print("Generating random 6-channel signal...")
        signal = np.random.randn(1000, 6).astype(np.float32)
        
    phonemes = engine.predict(signal)
    print(f"Predicted Phonemes: {phonemes}")
    
    # TTS Placeholder
    print("Synthesizing audio...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gru")
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()
    run_pipeline(args)
