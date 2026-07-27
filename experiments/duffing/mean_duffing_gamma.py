import os

import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_PATH = (
    "results/duffing/gamma_sweep/summary.csv"
)

OUTPUT_FOLDER = (
    "results/duffing/gamma_sweep/analysis"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True,
)


# ==========================================
# Load data
# ==========================================

df = pd.read_csv(SUMMARY_PATH)


# ==========================================
# Calculate statistics
# ==========================================

summary = (
    df
    .groupby("Gamma")
    .agg(
        MSE_Mean=("MSE", "mean"),
        MSE_Std=("MSE", "std"),
        MAE_Mean=("MAE", "mean"),
        MAE_Std=("MAE", "std"),
        Max_Error_Mean=(
            "Maximum Error",
            "mean",
        ),
        Max_Error_Std=(
            "Maximum Error",
            "std",
        ),
        R2_Mean=("R2 Score", "mean"),
        R2_Std=("R2 Score", "std"),
    )
    .reset_index()
)


# ==========================================
# Save mean summary
# ==========================================

summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "mean_summary.csv",
    ),
    index=False,
)

print(summary)


# ==========================================
# MSE plot
# ==========================================

plt.figure(figsize=(8, 5))

plt.errorbar(
    summary["Gamma"],
    summary["MSE_Mean"],
    yerr=summary["MSE_Std"],
    marker="o",
    capsize=5,
)

plt.xlabel("Forcing Amplitude (γ)")
plt.ylabel("Mean Squared Error (MSE)")

plt.title(
    "Neural Network Error vs Duffing Forcing Amplitude"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "mse_vs_duffing_gamma.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ==========================================
# R2 plot
# ==========================================

plt.figure(figsize=(8, 5))

plt.errorbar(
    summary["Gamma"],
    summary["R2_Mean"],
    yerr=summary["R2_Std"],
    marker="o",
    capsize=5,
)

plt.xlabel("Forcing Amplitude (γ)")
plt.ylabel("R² Score")

plt.title(
    "Neural Network Accuracy vs Duffing Forcing Amplitude"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "r2_vs_duffing_gamma.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print()
print("Analysis saved to:")
print(OUTPUT_FOLDER)