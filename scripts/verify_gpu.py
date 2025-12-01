import torch
import sys

def verify_gpu():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU Device Count: {device_count}")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    else:
        print("WARNING: CUDA is not available. Using CPU.")

if __name__ == "__main__":
    verify_gpu()
