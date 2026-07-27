import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Read mean summary
# ==========================================

df = pd.read_csv(
    "results/duffing/omega_sweep/mean_summary.csv"
)

# ==========================================
# MSE
# ==========================================

plt.figure(figsize=(8,5))

plt.errorbar(
    df["Omega"],
    df["MSE_Mean"],
    yerr=df["MSE_Std"],
    marker="o",
    capsize=5,
)

plt.xlabel("Duffing Omega")
plt.ylabel("Mean MSE")
plt.title("MSE vs Duffing Omega")

plt.grid(True)

plt.savefig(
    "results/duffing/omega_sweep/mse_vs_omega.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ==========================================
# R2
# ==========================================

plt.figure(figsize=(8,5))

plt.errorbar(
    df["Omega"],
    df["R2_Mean"],
    yerr=df["R2_Std"],
    marker="o",
    capsize=5,
)

plt.xlabel("Duffing Omega")
plt.ylabel("Mean R²")
plt.title("R² vs Duffing Omega")

plt.grid(True)

plt.savefig(
    "results/duffing/omega_sweep/r2_vs_omega.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Plots saved.")