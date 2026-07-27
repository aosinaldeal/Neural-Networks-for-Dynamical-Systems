import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# Paths
# ==========================================

INPUT_FILE = os.path.join(
    "results",
    "pendulum",
    "gravity_sweep",
    "summary.csv"
)

OUTPUT_FOLDER = os.path.join(
    "results",
    "pendulum",
    "gravity_sweep"
)


# ==========================================
# Load data
# ==========================================

df = pd.read_csv(INPUT_FILE)


# ==========================================
# Compute mean and std
# ==========================================

mean_df = (
    df.groupby("Gravity")
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
# Save CSV
# ==========================================

output_csv = os.path.join(
    OUTPUT_FOLDER,
    "mean_pendulum_gravity.csv"
)

mean_df.to_csv(
    output_csv,
    index=False
)


print(
    f"Saved: {output_csv}"
)


# ==========================================
# Plot function
# ==========================================

def save_plot(
    x,
    y,
    yerr,
    xlabel,
    ylabel,
    title,
    filename,
):

    plt.figure(figsize=(8,5))

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=5,
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(title)

    plt.grid(True)

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            filename
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================
# Create plots
# ==========================================

save_plot(
    mean_df["Gravity"],
    mean_df["MSE_Mean"],
    mean_df["MSE_Std"],
    "Gravity",
    "MSE",
    "MSE vs Gravity",
    "mse_vs_gravity.png",
)


save_plot(
    mean_df["Gravity"],
    mean_df["MAE_Mean"],
    mean_df["MAE_Std"],
    "Gravity",
    "MAE",
    "MAE vs Gravity",
    "mae_vs_gravity.png",
)


save_plot(
    mean_df["Gravity"],
    mean_df["Max_Error_Mean"],
    mean_df["Max_Error_Std"],
    "Gravity",
    "Maximum Error",
    "Maximum Error vs Gravity",
    "max_error_vs_gravity.png",
)


save_plot(
    mean_df["Gravity"],
    mean_df["R2_Mean"],
    mean_df["R2_Std"],
    "Gravity",
    "R² Score",
    "R² vs Gravity",
    "r2_vs_gravity.png",
)


print("Plots created successfully")