import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.gru.sensor_to_phoneme_gru import SensorToPhonemeGRU
from src.models.transformer.sensor_to_phoneme_transformer import SensorToPhonemeTransformer
from src.preprocessing.universal_preprocessing import UniversalPreprocessor
from src.utils.phoneme_utils import PhonemeTokenizer

def test_sanity():
    print("=== Starting Model Sanity Test ===")
    
    # 1. Test Preprocessing
    print("\n[1] Testing Universal Preprocessing...")
    raw_signal = np.random.randn(1000, 6) # 6 channels (AVC style)
    preprocessor = UniversalPreprocessor(target_rate=1000, fixed_length=512)
    processed = preprocessor.process(raw_signal, original_rate=1000)
    print(f"Raw shape: {raw_signal.shape} -> Processed shape: {processed.shape}")
    assert processed.shape == (512, 6)
    print("Preprocessing OK.")
    
    # 2. Test Tokenizer
    print("\n[2] Testing Tokenizer...")
    dummy_vocab = ["A", "B", "C"]
    tokenizer = PhonemeTokenizer(phoneme_list=dummy_vocab)
    encoded = tokenizer.encode(["A", "C", "B"])
    decoded = tokenizer.decode(encoded)
    print(f"Vocab: {dummy_vocab}")
    print(f"Encoded 'A C B': {encoded}")
    print(f"Decoded: {decoded}")
    assert decoded == ["A", "C", "B"]
    print("Tokenizer OK.")
    
    # 3. Test GRU Model (Dynamic Input)
    print("\n[3] Testing GRU Model (6 channels)...")
    input_dim = 6
    num_classes = len(tokenizer)
    model_gru = SensorToPhonemeGRU(input_dim=input_dim, num_classes=num_classes)
    
    input_tensor = torch.randn(2, 512, 6) # (B, T, C)
    output_gru, decoder_logits = model_gru(input_tensor) # (T, B, C) for CTC, (B, L, C) for Attn
    print(f"GRU Input: {input_tensor.shape}")
    print(f"GRU Output (CTC): {output_gru.shape}")
    print(f"GRU Output (Dec): {decoder_logits.shape}")
    
    # Expected output shape: (T/4, B, num_classes) due to pooling
    expected_T = 512 // 4
    assert output_gru.shape == (expected_T, 2, num_classes)
    print("GRU Model OK.")
    
    # 4. Test Transformer Model (Dynamic Input)
    print("\n[4] Testing Transformer Model (12 channels)...")
    input_dim_tr = 12
    model_tr = SensorToPhonemeTransformer(input_dim=input_dim_tr, num_classes=num_classes)
    
    input_tensor_tr = torch.randn(2, 100, 12)
    # Transformer needs target input for training forward pass
    tgt_tensor = torch.randint(0, num_classes, (2, 20)) # (B, T_tgt)
    
    output_tr = model_tr(input_tensor_tr, tgt_tensor)
    print(f"Transformer Input: {input_tensor_tr.shape}")
    print(f"Transformer Output: {output_tr.shape}")
    
    assert output_tr.shape == (2, 20, num_classes)
    print("Transformer Model OK.")

    print("\n[5] Testing Hybrid Forward/Backward...")
    # quick hybrid forward/backward test
    from src.losses.hybrid_loss import hybrid_loss
    from src.utils.config import Config
    import torch.nn as nn

    device = torch.device(Config.DEVICE)
    print(f"Testing on device: {device}")

    B, T, C = 2, 256, 6
    num_classes = 49
    x = torch.randn(B, T, C).to(device)
    input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
    # padded targets
    L = 30
    targets = torch.randint(1, num_classes, (B, L), dtype=torch.long).to(device)
    # ctc_targets flattened & lengths: create simple placeholders for test
    ctc_targets = torch.randint(1, num_classes, (B * L,), dtype=torch.long).to(device)
    ctc_input_lengths = torch.full((B,), T // 4, dtype=torch.long).to(device)  # account for pooling
    ctc_target_lengths = torch.full((B,), L, dtype=torch.long).to(device)

    model = SensorToPhonemeGRU(input_dim=C, num_classes=num_classes).to(device)
    ctc_logits, decoder_logits = model(x, x_lengths=input_lengths, targets=targets)
    
    if ctc_logits.dim() == 3 and decoder_logits.dim() == 3:
        print("Hybrid Output Shapes OK.")
    else:
        print(f"Shape Mismatch: CTC {ctc_logits.shape}, Dec {decoder_logits.shape}")

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    ctc_loss_val = ctc_loss_fn(ctc_logits, ctc_targets, ctc_input_lengths, ctc_target_lengths)
    attn_loss_val = ce_loss_fn(decoder_logits.view(-1, num_classes), targets.view(-1))
    loss = hybrid_loss(ctc_loss_val, attn_loss_val, lambda_ctc=0.6)
    loss.backward()
    print(f"Hybrid Backward Pass OK. Loss: {loss.item():.4f}")

    print("\n=== All Sanity Tests Passed! ===")

if __name__ == "__main__":
    test_sanity()
