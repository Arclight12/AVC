import os
import glob
import numpy as np
from .phoneme_utils import PhonemeTokenizer

def get_emg_uka_samples(root_dir):
    """
    Crawl EMG-UKA directory and return list of (adc_path, txt_path).
    """
    emg_dir = os.path.join(root_dir, "emg")
    align_dir = os.path.join(root_dir, "Alignments")
    
    samples = []
    
    # Recursive search for .adc files
    adc_files = []
    for root, dirs, files in os.walk(emg_dir):
        for file in files:
            if file.endswith(".adc"):
                adc_files.append(os.path.join(root, file))
    
    adc_files = sorted(adc_files)
    
    for adc_path in adc_files:
        basename = os.path.basename(adc_path)
        parts = basename.split('_')
        if len(parts) >= 4:
            # e07_002_001_0100.adc -> 002_001_0100
            id_str = "_".join(parts[1:]).replace(".adc", "")
            session = parts[1]
            block = parts[2]
            
            align_path = os.path.join(align_dir, session, block, f"phones_{id_str}.txt")
            
            if os.path.exists(align_path):
                samples.append((adc_path, align_path))
                
    return samples

def generate_synthetic_samples(num_samples=100, input_dim=7, max_len=100):
    """
    Generate synthetic samples (data, labels).
    """
    samples = []
    phonemes = ["A", "B", "C", "D", "E"] # Simple set
    
    for _ in range(num_samples):
        length = np.random.randint(20, max_len)
        data = np.random.randn(length, input_dim).astype(np.float32)
        
        label_len = np.random.randint(5, 20)
        label = [np.random.choice(phonemes) for _ in range(label_len)]
        
        samples.append((data, label))
        
    return samples, phonemes
