import os
import pandas as pd
import numpy as np
import json

# === CONFIG ===
AUG_RESULTS_DIR = r"D:\Thesis\aug_results"
OUTPUT_DIR = "D:/Thesis/outputs/aug_stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

target_cols = ["total_calories", "total_fat", "total_carb", "total_protein"]

def calculate_smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.divide(numerator, np.clip(denominator, 1e-6, None))) * 100
    return smape

# === Process each pair ===
files = sorted(os.listdir(AUG_RESULTS_DIR))
pairs = {}

# Group files into pairs based on shared prefix (excluding _predicted/_groundtruth)
for file in files:
    if file.endswith(".csv"):
        base = file.replace("_predicted.csv", "").replace("_groundtruth.csv", "")
        if base not in pairs:
            pairs[base] = {}
        if "_predicted" in file:
            pairs[base]["pred"] = file
        elif "_groundtruth" in file:
            pairs[base]["true"] = file

# Loop through each pair and compute stats
for level, files in pairs.items():
    if "true" not in files or "pred" not in files:
        print(f"Skipping incomplete pair for {level}")
        continue

    df_true = pd.read_csv(os.path.join(AUG_RESULTS_DIR, files["true"]))
    df_pred = pd.read_csv(os.path.join(AUG_RESULTS_DIR, files["pred"]))

    # Check alignment
    if not df_true["dish_id"].equals(df_pred["dish_id"]):
        print(f"Dish ID mismatch for {level}")
        continue

    stats = {}
    for col in target_cols:
        col_name = col.replace("total_", "")
        y_true = df_true[col].values
        y_pred = df_pred[col].values

        abs_errors = np.abs(y_true - y_pred)
        mae = np.mean(abs_errors)
        mae_pct = mae / (np.mean(np.abs(y_true)) + 1e-8) * 100
        smape = calculate_smape(y_true, y_pred)

        stats[f"{col_name}_MAE"] = round(mae, 2)
        stats[f"{col_name}_MAE_%"] = round(mae_pct, 2)
        stats[f"{col_name}_SMAPE"] = round(smape, 2)

    output_json = os.path.join(OUTPUT_DIR, f"{level}_stats.json")
    with open(output_json, "w") as f_out:
        json.dump(stats, f_out, indent=4)
    print(f"Saved stats for {level} → {output_json}")
