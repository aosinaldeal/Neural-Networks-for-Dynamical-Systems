import pandas as pd

# ==========================================
# Read summary
# ==========================================

df = pd.read_csv(
    "results/duffing/omega_sweep/summary.csv"
)

# ==========================================
# Compute statistics
# ==========================================

summary = (
    df
    .groupby("Omega")
    .agg({
        "MSE": ["mean", "std"],
        "MAE": ["mean", "std"],
        "Maximum Error": ["mean", "std"],
        "R2 Score": ["mean", "std"],
    })
)

summary.columns = [
    "MSE_Mean",
    "MSE_Std",
    "MAE_Mean",
    "MAE_Std",
    "Max_Error_Mean",
    "Max_Error_Std",
    "R2_Mean",
    "R2_Std",
]

summary = summary.reset_index()

summary.to_csv(
    "results/duffing/omega_sweep/mean_summary.csv",
    index=False,
)

print(summary)