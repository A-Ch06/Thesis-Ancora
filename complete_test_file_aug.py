import os
import pandas as pd

# ====== CONFIGURATION ======
SEG_TEST_DIR = r"D:\Thesis\outputs\aug_seg_test"
METADATA_PATH = r"D:\Thesis\outputs\test_dish_metadata_cleaned.csv"
OUTPUT_DIR = r"D:\Thesis\outputs\compl_aug_seg_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== LOAD FULL METADATA ======
metadata_df = pd.read_csv(METADATA_PATH)

# ====== PROCESS EACH SEGMENTATION TEST FILE ======
for file in os.listdir(SEG_TEST_DIR):
    if not file.endswith(".csv"):
        continue

    seg_path = os.path.join(SEG_TEST_DIR, file)
    seg_df = pd.read_csv(seg_path)

    # Ensure dish_id is string for match
    seg_df["dish_id"] = seg_df["dish_id"].astype(str)
    metadata_df["dish_id"] = metadata_df["dish_id"].astype(str)

    # Right join → keeps only rows from seg_df and pulls matching metadata
    merged_df = pd.merge(metadata_df, seg_df, on="dish_id", how="right")

    # Save output
    output_path = os.path.join(OUTPUT_DIR, file)
    merged_df.to_csv(output_path, index=False)
    print(f"Merged (right join): {file} → {output_path}")

print("\nAll files saved with metadata first, segmentation at the right.")
