import os

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# Paths
# ==========================================

SUMMARY_PATH = "results/duffing/summary.csv"
OUTPUT_FOLDER = "results/duffing/analysis"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True,
)


# ==========================================
# Load data
# ==========================================

df = pd.read_csv(SUMMARY_PATH)

# Remove exact duplicated experiments
df = df.drop_duplicates(
    subset=["Beta", "Random seed"]
)


# ==========================================
# Calculate statistics
# ==========================================

summary = (
    df
    .groupby("Beta")
    .agg(
        MSE_Mean=("MSE", "mean"),
        MSE_Std=("MSE", "std"),
        MAE_Mean=("MAE", "mean"),
        MAE_Std=("MAE", "std"),
        Max_Error_Mean=("Maximum Error", "mean"),
        Max_Error_Std=("Maximum Error", "std"),
        R2_Mean=("R2 Score", "mean"),
        R2_Std=("R2 Score", "std"),
    )
    .reset_index()
)

print(summary)

# ==========================================
# Save statistics
# ==========================================

MEAN_PATH = os.path.join(
    OUTPUT_FOLDER,
    "mean_summary.csv",
)

summary.to_csv(
    MEAN_PATH,
    index=False,
)

print()
print(
    f"Mean summary saved to: {MEAN_PATH}"
)


# ==========================================
# MSE vs Duffing Beta
# ==========================================

plt.figure(figsize=(8, 5))

plt.errorbar(
    summary["Beta"],
    summary["MSE_Mean"],
    yerr=summary["MSE_Std"],
    marker="o",
    capsize=5,
)

plt.xlabel("Duffing Nonlinearity Strength (β)")
plt.ylabel("Mean Squared Error (MSE)")

plt.title(
    "Neural Network Error vs Duffing Nonlinearity"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "mse_vs_duffing_beta.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ==========================================
# R2 vs Duffing Beta
# ==========================================

plt.figure(figsize=(8, 5))

plt.errorbar(
    summary["Beta"],
    summary["R2_Mean"],
    yerr=summary["R2_Std"],
    marker="o",
    capsize=5,
)

plt.xlabel("Duffing Nonlinearity Strength (β)")
plt.ylabel("R² Score")

plt.title(
    "Neural Network Accuracy vs Duffing Nonlinearity"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "r2_vs_duffing_beta.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print()
print("Plots saved to:")
print(OUTPUT_FOLDER)