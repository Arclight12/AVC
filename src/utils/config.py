
class Config:
    # Data
    TARGET_SAMPLING_RATE = 1000
    WINDOW_SIZE = 256 # For windowing if needed
    
    # Model - GRU
    GRU_HIDDEN_DIM = 512
    GRU_NUM_LAYERS = 4
    GRU_DROPOUT = 0.3
    
    # Model - Transformer
    TRANSFORMER_D_MODEL = 256
    TRANSFORMER_NHEAD = 4
    TRANSFORMER_NUM_ENCODER_LAYERS = 3
    TRANSFORMER_NUM_DECODER_LAYERS = 3
    TRANSFORMER_DIM_FEEDFORWARD = 512
    TRANSFORMER_DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    EPOCHS = 10
    
    # Device
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Hybrid Loss
    USE_HYBRID_LOSS = True
    USE_ATTENTION_DECODER = True
    LAMBDA_START = 0.7
    LAMBDA_MID = 0.5
    LAMBDA_END = 0.3
    GRAD_CLIP = 5.0
    
    # Token IDs (Must match PhonemeTokenizer)
    CTC_BLANK_ID = 0
    PAD_ID = 0
    SOS_ID = 1
    EOS_ID = 2
    
    # Paths (Default)
    DATA_DIR = "d:/AVC/data/archive/EMG-UKA-Trial-Corpus"
    
    # Defaults for Scripts
    DEFAULT_SYNTH_EPOCHS = 50
    DEFAULT_EMG_UKA_EPOCHS = 50
    DEFAULT_EMG_UKA_PATH = "d:/AVC/data/archive/EMG-UKA-Trial-Corpus"
    
    # Phonemes
    # We can load this dynamically, but defaults here
    SPECIAL_TOKENS = ["<BLANK>", "<SOS>", "<EOS>"]
