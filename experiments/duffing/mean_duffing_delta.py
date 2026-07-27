import pandas as pd

# ==========================================
# Load summary
# ==========================================

df = pd.read_csv("results/duffing/delta_sweep/summary.csv")

# ==========================================
# Compute statistics
# ==========================================

mean_df = (
    df.groupby("Delta")
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

# ==========================================
# Save
# ==========================================

mean_df.to_csv(
    "results/duffing/delta_sweep/mean.csv",
    index=False,
)

print(mean_df)


import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Load data
# ==========================================

df = pd.read_csv("results/duffing/delta_sweep/mean.csv")

# ==========================================
# MSE vs Delta
# ==========================================

plt.figure(figsize=(8,5))

plt.errorbar(
    df["Delta"],
    df["MSE_Mean"],
    yerr=df["MSE_Std"],
    marker="o",
    capsize=5,
)

plt.title("MSE vs Duffing Delta")
plt.xlabel("Delta")
plt.ylabel("Mean Squared Error")

plt.grid(True)

plt.savefig(
    "results/duffing/delta_sweep/mse_vs_delta.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ==========================================
# R² vs Delta
# ==========================================

plt.figure(figsize=(8,5))

plt.errorbar(
    df["Delta"],
    df["R2_Mean"],
    yerr=df["R2_Std"],
    marker="o",
    capsize=5,
)

plt.title("R² Score vs Duffing Delta")
plt.xlabel("Delta")
plt.ylabel("R² Score")

plt.grid(True)

plt.savefig(
    "results/duffing/delta_sweep/r2_vs_delta.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Graphs saved successfully.")