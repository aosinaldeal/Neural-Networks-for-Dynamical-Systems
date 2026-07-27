import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# Load summary
# ==========================================

df = pd.read_csv(
    "results/pendulum/length_sweep/summary.csv"
)


# ==========================================
# Mean and standard deviation
# ==========================================

mean_df = (
    df
    .groupby("Length")
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

output_folder = "results/pendulum/length_sweep"

mean_df.to_csv(
    f"{output_folder}/mean_length.csv",
    index=False,
)

print(mean_df)


# ==========================================
# Plot function
# ==========================================

def create_plot(
    x,
    y,
    std,
    xlabel,
    ylabel,
    title,
    filename,
):

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        x,
        y,
        yerr=std,
        marker="o",
        capsize=5,
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/{filename}"
    )

    plt.close()


# ==========================================
# MSE
# ==========================================

create_plot(
    mean_df["Length"],
    mean_df["MSE_Mean"],
    mean_df["MSE_Std"],
    "Pendulum Length (m)",
    "Mean MSE",
    "MSE vs Pendulum Length",
    "mse_vs_length.png",
)


# ==========================================
# MAE
# ==========================================

create_plot(
    mean_df["Length"],
    mean_df["MAE_Mean"],
    mean_df["MAE_Std"],
    "Pendulum Length (m)",
    "Mean MAE",
    "MAE vs Pendulum Length",
    "mae_vs_length.png",
)


# ==========================================
# Maximum Error
# ==========================================

create_plot(
    mean_df["Length"],
    mean_df["Max_Error_Mean"],
    mean_df["Max_Error_Std"],
    "Pendulum Length (m)",
    "Mean Maximum Error",
    "Maximum Error vs Pendulum Length",
    "max_error_vs_length.png",
)


# ==========================================
# R2
# ==========================================

create_plot(
    mean_df["Length"],
    mean_df["R2_Mean"],
    mean_df["R2_Std"],
    "Pendulum Length (m)",
    "Mean R²",
    "R² vs Pendulum Length",
    "r2_vs_length.png",
)


print()
print("Finished.")