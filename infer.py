import argparse
from src.inference.infer_pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gru", choices=["gru", "transformer"])
    parser.add_argument("--sentence", type=str, default="WHAT IS YOUR NAME")
    args = parser.parse_args()
    run_pipeline(args)
