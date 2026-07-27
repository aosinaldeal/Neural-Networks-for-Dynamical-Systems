import os
import re
import pandas as pd


# ==========================================
# Paths
# ==========================================

RESULTS_PATH = os.path.join(
    "results",
    "duffing",
    "alpha_sweep"
)


OUTPUT_FILE = os.path.join(
    RESULTS_PATH,
    "mean.csv"
)


# ==========================================
# Extract metrics
# ==========================================

def extract_metric(file, name):

    with open(file, "r") as f:
        text = f.read()

    match = re.search(
        rf"{name}: ([0-9.eE+-]+)",
        text
    )

    if match:
        return float(match.group(1))

    return None



# ==========================================
# Read experiments
# ==========================================

results = []


for folder in os.listdir(RESULTS_PATH):

    folder_path = os.path.join(
        RESULTS_PATH,
        folder
    )


    if not os.path.isdir(folder_path):
        continue


    metrics_file = os.path.join(
        folder_path,
        "metrics.txt"
    )


    if not os.path.exists(metrics_file):
        continue


    # Extract alpha from folder name

    alpha_match = re.search(
        r"alpha_([0-9.]+)",
        folder
    )


    if alpha_match is None:
        continue


    alpha = float(
        alpha_match.group(1)
    )


    mse = extract_metric(
        metrics_file,
        "MSE"
    )

    mae = extract_metric(
        metrics_file,
        "MAE"
    )

    max_error = extract_metric(
        metrics_file,
        "Maximum Error"
    )

    r2 = extract_metric(
        metrics_file,
        "R2 Score"
    )


    results.append(
        {
            "Alpha": alpha,
            "MSE": mse,
            "MAE": mae,
            "Maximum Error": max_error,
            "R2 Score": r2,
        }
    )



df = pd.DataFrame(results)


# ==========================================
# Calculate mean/std
# ==========================================


summary = (
    df
    .groupby("Alpha")
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



summary.to_csv(
    OUTPUT_FILE,
    index=False
)


print("="*50)
print("Alpha summary created")
print("="*50)

print(summary)

print()
print(
    f"Saved to: {OUTPUT_FILE}"
)




#plot

import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# Paths
# ==========================================


PATH = os.path.join(
    "results",
    "duffing",
    "alpha_sweep"
)


CSV_FILE = os.path.join(
    PATH,
    "summary.csv"
)


df = pd.read_csv(
    CSV_FILE
)


# ==========================================
# Plot function
# ==========================================


def plot_metric(
    mean,
    std,
    ylabel,
    filename
):

    plt.figure(
        figsize=(8,5)
    )


    plt.errorbar(
        df["Alpha"],
        df[mean],
        yerr=df[std],
        marker="o",
        capsize=5,
    )


    plt.xlabel(
        "Alpha"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        f"Duffing Oscillator - {ylabel} vs Alpha"
    )

    plt.grid(True)


    plt.savefig(
        os.path.join(
            PATH,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



# ==========================================
# Generate plots
# ==========================================


plot_metric(
    "MSE_Mean",
    "MSE_Std",
    "MSE",
    "alpha_mse.png"
)


plot_metric(
    "MAE_Mean",
    "MAE_Std",
    "MAE",
    "alpha_mae.png"
)


plot_metric(
    "Max_Error_Mean",
    "Max_Error_Std",
    "Maximum Error",
    "alpha_max_error.png"
)


plot_metric(
    "R2_Mean",
    "R2_Std",
    "R² Score",
    "alpha_r2.png"
)


print("Alpha plots generated")