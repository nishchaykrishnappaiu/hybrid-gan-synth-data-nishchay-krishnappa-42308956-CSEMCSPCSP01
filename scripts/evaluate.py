import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp
from scripts.utils import detect_column_types

def _to_numeric(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")

def evaluate(real_csv: str, synthetic_csv: str, plots_dir: str):
    real = pd.read_csv(real_csv)
    synth = pd.read_csv(synthetic_csv)
    common_cols = [c for c in real.columns if c in synth.columns]
    real = real[common_cols].copy()
    synth = synth[common_cols].copy()
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    report_lines = []
    for col in common_cols:
        r = real[col]
        s = synth[col]
        r_num = _to_numeric(r)
        s_num = _to_numeric(s)

        r_num_valid = r_num.dropna()
        s_num_valid = s_num.dropna()

        numeric_possible = (r_num_valid.size > 0) and (s_num_valid.size > 0)

        if numeric_possible and (r_num_valid.nunique() > 1 or s_num_valid.nunique() > 1):
            ks_stat, p_val = ks_2samp(r_num_valid, s_num_valid)
            report_lines.append(f"CONTINUOUS {col}: KS={ks_stat:.4f}, p={p_val:.4f}")

            plt.figure(figsize=(6,4))
            plt.hist(r_num_valid, bins=30, alpha=0.5, label="Real")
            plt.hist(s_num_valid, bins=30, alpha=0.5, label="Synthetic")
            plt.title(f"Distribution - {col}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(plots_dir)/f"hist_{col}.png", dpi=150)
            plt.close()
        else:
            r_counts = r.astype(str).value_counts(normalize=True)
            s_counts = s.astype(str).value_counts(normalize=True)
            keys = sorted(set(r_counts.index).union(set(s_counts.index)))
            df_plot = pd.DataFrame({
                "Real": [r_counts.get(k, 0.0) for k in keys],
                "Synthetic": [s_counts.get(k, 0.0) for k in keys]
            }, index=keys)

            ax = df_plot.plot(kind="bar", figsize=(8,4))
            ax.set_title(f"Category Distribution - {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Proportion")
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(Path(plots_dir)/f"bar_{col}.png", dpi=150)
            plt.close(fig)

            l1 = (df_plot["Real"] - df_plot["Synthetic"]).abs().sum()
            report_lines.append(f"CATEGORICAL {col}: L1 distance={l1:.4f}")

    with open(Path(plots_dir) / "evaluation_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print("✅ Saved plots to:", plots_dir)
    print("✅ Saved report:", str(Path(plots_dir) / "evaluation_report.txt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_csv", required=True)
    parser.add_argument("--synthetic_csv", required=True)
    parser.add_argument("--plots_dir", required=True)
    args = parser.parse_args()
    evaluate(args.real_csv, args.synthetic_csv, args.plots_dir)
