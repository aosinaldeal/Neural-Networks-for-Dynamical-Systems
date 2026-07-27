import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "forced_harmonic"
    / "aggregated_summary.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "forced_harmonic"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

forcing_omega = df["Forcing Omega"]


# ------------------------------------------------------------
# MSE Mean vs Forcing Omega
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.errorbar(
    forcing_omega,
    df["MSE_Mean"],
    yerr=df["MSE_Std"],
    marker="o",
    capsize=4,
)

plt.xlabel("Forcing Frequency $\\omega_f$")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Prediction Error vs Forcing Frequency")

plt.yscale("log")

plt.grid(True, alpha=0.3)
plt.tight_layout()

mse_path = OUTPUT_DIR / "mse_vs_forcing_omega.png"

plt.savefig(
    mse_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# R² Mean vs Forcing Omega
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.errorbar(
    forcing_omega,
    df["R2_Mean"],
    yerr=df["R2_Std"],
    marker="o",
    capsize=4,
)

plt.xlabel("Forcing Frequency $\\omega_f$")
plt.ylabel("$R^2$ Score")
plt.title("$R^2$ Score vs Forcing Frequency")

plt.grid(True, alpha=0.3)
plt.tight_layout()

r2_path = OUTPUT_DIR / "r2_vs_forcing_omega.png"

plt.savefig(
    r2_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# Done
# ------------------------------------------------------------

print("Plots generated successfully:")
print(mse_path)
print(r2_path)