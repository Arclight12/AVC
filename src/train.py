import argparse
import torch
from torch.utils.data import DataLoader
from src.utils.config import Config
from src.utils.phoneme_utils import PhonemeTokenizer
from src.utils.data_helpers import get_emg_uka_samples, generate_synthetic_samples
from src.utils import EMG_UKA_PHONEMES
from src.preprocessing.universal_preprocessing import UniversalPreprocessor
from src.dataset.biosignal_dataset import BiosignalDataset, collate_fn_biosignal
from src.models.gru.sensor_to_phoneme_gru import SensorToPhonemeGRU
from src.models.transformer.sensor_to_phoneme_transformer import SensorToPhonemeTransformer
from src.training.trainer import Trainer

def main(args):
    # Device
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Config & Tokenizer
    phoneme_list = list(EMG_UKA_PHONEMES.keys())
    tokenizer = PhonemeTokenizer(phoneme_list=phoneme_list)
    
    # 2. Data
    if args.dataset_type == "emg_uka":
        if not args.data_dir:
            raise ValueError("Must provide --data_dir for emg_uka")
        samples = get_emg_uka_samples(args.data_dir)
    elif args.dataset_type == "synthetic":
        samples, _ = generate_synthetic_samples(num_samples=args.num_samples, input_dim=7)
    else:
        raise ValueError("Unknown dataset type")
        
    preprocessor = UniversalPreprocessor(target_rate=Config.TARGET_SAMPLING_RATE)
    dataset = BiosignalDataset(samples, tokenizer, preprocessor, augment=args.augment)
    
    # Split Dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_biosignal)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_biosignal)
    
    # 3. Model
    dummy_signal, _ = dataset[0]
    input_dim = dummy_signal.shape[1]
    num_classes = len(tokenizer)
    print(f"Input Dim: {input_dim}, Num Classes: {num_classes}")
    
    if args.model_type == "gru":
        model = SensorToPhonemeGRU(input_dim, num_classes, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
    else:
        model = SensorToPhonemeTransformer(input_dim, num_classes)
        
    model = model.to(device)

    # 4. Train
    trainer = Trainer(model, train_loader, val_loader, tokenizer, Config, device, dataset_type=args.dataset_type)
    
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
        trainer.validate()
        
    trainer.save_checkpoint(f"model_{args.model_type}_{args.dataset_type}.pth")
    print(f"Model saved to model_{args.model_type}_{args.dataset_type}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gru", choices=["gru", "transformer"])
    parser.add_argument("--dataset_type", type=str, default="synthetic", choices=["synthetic", "emg_uka"])
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=Config.GRU_HIDDEN_DIM)
    parser.add_argument("--num_layers", type=int, default=Config.GRU_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=Config.GRU_DROPOUT)
    args = parser.parse_args()
    main(args)
