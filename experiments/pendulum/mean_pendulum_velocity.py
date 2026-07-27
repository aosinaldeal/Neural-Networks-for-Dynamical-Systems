import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# Paths
# ==========================================

INPUT_FILE = os.path.join(
    "results",
    "pendulum",
    "velocity_sweep",
    "summary.csv",
)

OUTPUT_FOLDER = os.path.join(
    "results",
    "pendulum",
    "velocity_sweep",
)


# ==========================================
# Load data
# ==========================================

df = pd.read_csv(INPUT_FILE)


# ==========================================
# Compute mean and std
# ==========================================

mean_df = (
    df
    .groupby("Initial Angular Velocity")
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
# Save mean.csv
# ==========================================

mean_path = os.path.join(
    OUTPUT_FOLDER,
    "mean.csv",
)

mean_df.to_csv(
    mean_path,
    index=False,
)


print(
    f"Saved: {mean_path}"
)


# ==========================================
# Plot function
# ==========================================

def create_plot(
    x,
    y,
    xlabel,
    ylabel,
    title,
    filename,
):

    plt.figure(figsize=(8,5))

    plt.plot(
        x,
        y,
        marker="o",
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(title)

    plt.grid(True)

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            filename,
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================
# Create plots
# ==========================================

velocity = mean_df[
    "Initial Angular Velocity"
]


create_plot(
    velocity,
    mean_df["MSE_Mean"],
    "Initial Angular Velocity",
    "Mean MSE",
    "MSE vs Initial Angular Velocity",
    "mse_vs_velocity.png",
)


create_plot(
    velocity,
    mean_df["MAE_Mean"],
    "Initial Angular Velocity",
    "Mean MAE",
    "MAE vs Initial Angular Velocity",
    "mae_vs_velocity.png",
)


create_plot(
    velocity,
    mean_df["Max_Error_Mean"],
    "Initial Angular Velocity",
    "Mean Maximum Error",
    "Maximum Error vs Initial Angular Velocity",
    "max_error_vs_velocity.png",
)


create_plot(
    velocity,
    mean_df["R2_Mean"],
    "Initial Angular Velocity",
    "Mean R2 Score",
    "R2 Score vs Initial Angular Velocity",
    "r2_vs_velocity.png",
)


print("Plots generated successfully")