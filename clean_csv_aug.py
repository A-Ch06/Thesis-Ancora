import os
import pandas as pd

# ====== CONFIGURATION ======
INPUT_DIR = r"D:\Thesis\outputs\segmentation_areas"
OUTPUT_DIR = r"D:\Thesis\outputs\aug_segmentation_areas_cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== NON-FOOD COLUMNS TO REMOVE ======
NON_FOOD_COLUMNS = {
    "background",
    "food_containers",
    "dining_tools",
}

# ====== PROCESS ALL CSV FILES ======
for csv_file in os.listdir(INPUT_DIR):
    if not csv_file.endswith(".csv"):
        continue

    input_path = os.path.join(INPUT_DIR, csv_file)
    df = pd.read_csv(input_path)

    # Extract dish_id from filename (e.g., "dish_1556572657_rgb.png" → "dish_1556572657")
    if "filename" in df.columns:
        df["dish_id"] = df["filename"].apply(lambda x: x.split("_rgb")[0])
        df = df.drop(columns=["filename"])  # Drop original filename column

    # Drop non-food columns
    df = df[[col for col in df.columns if col not in NON_FOOD_COLUMNS]]

    # Reorder dish_id first
    cols = df.columns.tolist()
    if "dish_id" in cols:
        cols = ["dish_id"] + [c for c in cols if c != "dish_id"]
        df = df[cols]

    # Save cleaned file
    output_path = os.path.join(OUTPUT_DIR, csv_file)
    df.to_csv(output_path, index=False)
    print(f"Cleaned: {csv_file} → {output_path}")

print("\nAll CSVs cleaned: non-food columns removed, filename dropped, dish_id corrected.")
