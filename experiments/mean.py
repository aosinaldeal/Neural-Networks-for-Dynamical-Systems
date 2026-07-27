from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results/damped_harmonic")
OUTPUT_FILE = Path("results/mean.csv")

rows = []

for experiment_dir in RESULTS_DIR.iterdir():

    if not experiment_dir.is_dir():
        continue

    metrics_file = experiment_dir / "metrics.txt"

    if not metrics_file.exists():
        continue

    beta = None
    seed = None
    mse = None
    mae = None
    max_error = None

    with open(metrics_file, "r") as f:

        for line in f:

            line = line.strip()

            if line.startswith("Beta:"):
                beta = float(line.split(":")[1].strip())

            elif line.startswith("Random Seed:"):
                seed = int(line.split(":")[1].strip())

            elif line.startswith("MSE:"):
                mse = float(line.split(":")[1].strip())

            elif line.startswith("MAE:"):
                mae = float(line.split(":")[1].strip())

            elif line.startswith("Maximum Error:"):
                max_error = float(line.split(":")[1].strip())

    rows.append({
        "Beta": beta,
        "Seed": seed,
        "MSE": mse,
        "MAE": mae,
        "Maximum Error": max_error,
    })

df = pd.DataFrame(rows)

summary = (
    df.groupby("Beta")
      .agg(
          Experiments=("Seed", "count"),
          MSE_Mean=("MSE", "mean"),
          MSE_STD=("MSE", "std"),
          MAE_Mean=("MAE", "mean"),
          MAE_STD=("MAE", "std"),
          MaxError_Mean=("Maximum Error", "mean"),
          MaxError_STD=("Maximum Error", "std"),
      )
      .reset_index()
      .sort_values("Beta")
)

summary = summary.round(6)

summary.to_csv(OUTPUT_FILE, index=False)

import matplotlib.pyplot as plt

# ----------------------------------------
# MSE
# ----------------------------------------

plt.figure(figsize=(6,4))

plt.errorbar(
    summary["Beta"],
    summary["MSE_Mean"],
    yerr=summary["MSE_STD"],
    marker="o",
    capsize=5
)

plt.xlabel("Damping coefficient (β)")
plt.ylabel("Mean MSE")
plt.title("Model performance vs damping coefficient")

plt.grid(True)

plt.tight_layout()

plt.savefig("results/mse_vs_beta.png", dpi=300)

plt.close()

plt.figure(figsize=(6,4))

plt.errorbar(
    summary["Beta"],
    summary["MAE_Mean"],
    yerr=summary["MAE_STD"],
    marker="o",
    capsize=5
)

plt.xlabel("Damping coefficient (β)")
plt.ylabel("Mean MAE")
plt.title("Mean Absolute Error vs damping coefficient")

plt.grid(True)

plt.tight_layout()

plt.savefig("results/mae_vs_beta.png", dpi=300)

plt.close()

# ----------------------------------------
# Maximum Error
# ----------------------------------------

plt.figure(figsize=(6,4))

plt.errorbar(
    summary["Beta"],
    summary["MaxError_Mean"],
    yerr=summary["MaxError_STD"],
    marker="o",
    capsize=5
)

plt.xlabel("Damping coefficient (β)")
plt.ylabel("Mean Maximum Error")
plt.title("Maximum Error vs damping coefficient")

plt.grid(True)

plt.tight_layout()

plt.savefig("results/max_error_vs_beta.png", dpi=300)

plt.close()

print("\nSummary:\n")
print(summary)

print(f"\nSaved to: {OUTPUT_FILE}")