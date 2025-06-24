import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# ==== CONFIG ====
AUG_TEST_DIR = r"D:\Thesis\outputs\compl_aug_seg_test"
BEST_MODEL_PATH = "D:/Thesis/models/best_nutrient_predictor.keras"
PLOTS_DIR = "D:/Thesis/aug_plots"
RESULTS_DIR = "D:/Thesis/aug_results"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==== CONSTANTS ====
target_cols = ["total_calories", "total_fat", "total_carb", "total_protein"]
ignore_cols = ["dish_id", "ingredients", "total_mass", "other_food"] + target_cols

# ==== SCATTER PLOT ====
def save_scatter(y_true, y_pred, nutrient, aug_name):
    folder = os.path.join(PLOTS_DIR, aug_name)
    os.makedirs(folder, exist_ok=True)
    i = target_cols.index(nutrient)
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, edgecolor='k')
    max_val = max(np.max(y_true[:, i]), np.max(y_pred[:, i]))
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel(f"Actual {nutrient}")
    plt.ylabel(f"Predicted {nutrient}")
    plt.title(f"{aug_name} - {nutrient}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{nutrient.lower()}_scatter.png"), dpi=300)
    plt.close()

# ==== LOAD TRAINED MODEL ====
model = tf.keras.models.load_model(BEST_MODEL_PATH)
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==== PROCESS EACH AUGMENTED CSV ====
for filename in os.listdir(AUG_TEST_DIR):
    if not filename.endswith(".csv"):
        continue

    aug_path = os.path.join(AUG_TEST_DIR, filename)
    aug_name = os.path.splitext(filename)[0]

    df = pd.read_csv(aug_path)
    segmentation_cols = [col for col in df.columns if col not in ignore_cols]

    X_seg_test = df[segmentation_cols].values.astype(np.float32)
    X_bert = bert_model.encode(df["ingredients"].tolist(), show_progress_bar=True).astype(np.float32)
    X_test = np.hstack((X_seg_test, X_bert))
    y_test = df[target_cols].values.astype(np.float32)

    # Predict
    y_pred = model.predict(X_test)

    # Save scatter plots
    for nutrient in target_cols:
        save_scatter(y_test, y_pred, nutrient, aug_name)

    # Save predicted and actual CSVs
    pred_df = pd.DataFrame(y_pred, columns=target_cols)
    pred_df.insert(0, "dish_id", df["dish_id"])
    pred_df.to_csv(os.path.join(RESULTS_DIR, f"{aug_name}_predicted.csv"), index=False)

    truth_df = pd.DataFrame(y_test, columns=target_cols)
    truth_df.insert(0, "dish_id", df["dish_id"])
    truth_df.to_csv(os.path.join(RESULTS_DIR, f"{aug_name}_groundtruth.csv"), index=False)

print("All augmented test sets processed.")
