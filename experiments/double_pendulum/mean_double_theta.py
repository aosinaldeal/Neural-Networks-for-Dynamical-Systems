import os
import pandas as pd
import matplotlib.pyplot as plt
import config


# ==========================================
# Paths
# ==========================================

SWEEP_NAME = "length2"

input_file = os.path.join(
    "results",
    "double_pendulum",
    SWEEP_NAME,
    "summary.csv"
)

output_folder = os.path.join(
    "results",
    "double_pendulum",
    SWEEP_NAME
)


os.makedirs(
    output_folder,
    exist_ok=True
)


# ==========================================
# Read summary
# ==========================================

df = pd.read_csv(input_file)


# ==========================================
# Group by parameter
# ==========================================

parameter = "Length 2"


mean_df = (
    df
    .groupby(parameter)
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


# Replace NaN std (only one seed)
mean_df = mean_df.fillna(0)


# ==========================================
# Save csv
# ==========================================

csv_path = os.path.join(
    output_folder,
    f"{SWEEP_NAME}.csv"
)

mean_df.to_csv(
    csv_path,
    index=False
)


print(
    f"Saved: {csv_path}"
)


# ==========================================
# MSE plot
# ==========================================

plt.figure(figsize=(8,5))


plt.errorbar(
    mean_df[parameter],
    mean_df["MSE_Mean"],
    yerr=mean_df["MSE_Std"],
    marker="o",
)


plt.xlabel(
    "Initial Theta 1 (rad)"
)

plt.ylabel(
    "MSE"
)

plt.title(
    "Double Pendulum - MSE vs Initial Theta 1"
)

plt.grid(True)


plt.savefig(
    os.path.join(
        output_folder,
        "mse_vs_theta1.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# ==========================================
# R2 plot
# ==========================================

plt.figure(figsize=(8,5))


plt.errorbar(
    mean_df[parameter],
    mean_df["R2_Mean"],
    yerr=mean_df["R2_Std"],
    marker="o",
)


plt.xlabel(
    "Initial Theta 1 (rad)"
)

plt.ylabel(
    "R² Score"
)

plt.title(
    "Double Pendulum - R² vs Initial Theta 1"
)

plt.grid(True)


plt.savefig(
    os.path.join(
        output_folder,
        "r2_vs_theta1.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("Plots generated successfully")