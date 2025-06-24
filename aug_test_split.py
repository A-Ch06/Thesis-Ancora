import os
import pandas as pd

# ====== CONFIGURATION ======
CLEANED_AUG_DIR = r"D:\Thesis\outputs\aug_segmentation_areas_cleaned"
OUTPUT_TEST_DIR = r"D:\Thesis\outputs\aug_seg_test"
TEST_IDS_PATH = r"D:\Thesis\nutrition5k_data\nutrition5k_dataset\dish_ids\splits\rgb_test_ids.txt"

os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)

# ====== LOAD TEST IDs ======
with open(TEST_IDS_PATH, 'r') as f:
    test_ids = set(line.strip() for line in f)

# ====== PROCESS EACH CLEANED CSV ======
for filename in os.listdir(CLEANED_AUG_DIR):
    if not filename.endswith(".csv"):
        continue

    input_path = os.path.join(CLEANED_AUG_DIR, filename)
    df = pd.read_csv(input_path)

    # Filter only test dish IDs
    df_test = df[df["dish_id"].astype(str).isin(test_ids)]

    # Save result
    output_path = os.path.join(OUTPUT_TEST_DIR, filename)
    df_test.to_csv(output_path, index=False)
    print(f"Saved test split: {filename} → {output_path}")

print("\nAll augmentation test splits saved.")
