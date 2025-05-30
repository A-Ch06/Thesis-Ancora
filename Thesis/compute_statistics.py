"""
Compute MAE, %MAE, and SMAPE between groundtruth and predicted nutrition values.

Assumes CSVs contain:
- 'dish_id', 'total_calories', 'total_fat', 'total_carb', 'total_protein'
- Optional: 'total_mass'

Output is a JSON file with metrics per nutrient.
"""

import pandas as pd
import numpy as np
import json

# ==== File paths ====
groundtruth_csv = "D:\Thesis\groundtruth_test_values.csv"
predicted_csv = "D:\Thesis\predicted_test_values.csv"
output_json = "D:/Thesis/outputs/output_statistics.json"

# ==== Columns to evaluate ====
target_cols = ["total_calories", "total_fat", "total_carb", "total_protein"]

# ==== Read data ====
df_true = pd.read_csv(groundtruth_csv)
df_pred = pd.read_csv(predicted_csv)

# ==== Check matching dish_ids ====
if not df_true["dish_id"].equals(df_pred["dish_id"]):
    raise ValueError("Dish IDs do not match or are not in the same order.")

# ==== Compute statistics ====
stats = {}

def calculate_smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.divide(numerator, np.clip(denominator, 1e-6, None))) * 100
    return smape

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

# ==== Save to JSON ====
with open(output_json, "w") as f_out:
    json.dump(stats, f_out, indent=4)
print(f"Saved statistics (including SMAPE) to {output_json}")
