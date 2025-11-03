import argparse
import joblib
import pandas as pd
from pathlib import Path
import torch

def generate(model_path: str, n_rows: int, output_csv: str):
    model = joblib.load(model_path)
    use_gpu = torch.cuda.is_available()

    # If the model exposes a device toggle, try to move it
    for attr, val in (("device", "cuda" if use_gpu else "cpu"), ("cuda", use_gpu), ("enable_gpu", use_gpu)):
        if hasattr(model, attr):
            try:
                setattr(model, attr, val)
            except Exception:
                pass

    samples = None
    for method in ("sample", "generate", "sample_rows", "sample_table"):
        if hasattr(model, method):
            fn = getattr(model, method)
            try:
                samples = fn(n_rows)
            except TypeError:
                samples = fn(num_rows=n_rows)
            break

    if samples is None:
        raise RuntimeError("No compatible sampling method found on the model.")

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
