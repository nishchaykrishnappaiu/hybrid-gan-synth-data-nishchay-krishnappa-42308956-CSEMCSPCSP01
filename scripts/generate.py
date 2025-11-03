import argparse
import joblib
import pandas as pd
from pathlib import Path

def generate(model_path: str, n_rows: int, output_csv: str):
    ctgan = joblib.load(model_path)
    samples = ctgan.sample(n_rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_csv, index=False)
    print(f"✅ Generated {len(samples)} rows -> {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_rows", type=int, required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    generate(args.model_path, args.n_rows, args.output_csv)
