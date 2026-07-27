import pandas as pd

INPUT_FILE = "results/forced_harmonic/summary.csv"
OUTPUT_FILE = "results/forced_harmonic/aggregated_summary.csv"


df = pd.read_csv(INPUT_FILE)

# Remove duplicated experiments
df = df.drop_duplicates(
    subset=[
        "Random seed",
        "Forcing Omega",
        "Forcing Amplitude",
        "Amplitude",
        "Omega",
        "Beta",
        "Phase",
    ]
)

aggregated = (
    df.groupby("Forcing Omega")
    .agg(
        MSE_Mean=("MSE", "mean"),
        MSE_Std=("MSE", "std"),
        MAE_Mean=("MAE", "mean"),
        MAE_Std=("MAE", "std"),
        Max_Error_Mean=("Maximum Error", "mean"),
        Max_Error_Std=("Maximum Error", "std"),
        R2_Mean=("R2 Score", "mean"),
        R2_Std=("R2 Score", "std"),
        Num_Runs=("Random seed", "count"),
    )
    .reset_index()
)

aggregated.to_csv(OUTPUT_FILE, index=False)

print(aggregated)
print()
print(f"Aggregated summary saved to: {OUTPUT_FILE}")