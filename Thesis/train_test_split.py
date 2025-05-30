import pandas as pd

# Paths
cleaned_csv = "D:\Thesis\outputs\segmentation_areas_cleaned.csv"
train_ids_path = "D:\\Thesis\\nutrition5k_data\\nutrition5k_dataset\\dish_ids\\splits\\rgb_train_ids.txt"
test_ids_path = "D:\\Thesis\\nutrition5k_data\\nutrition5k_dataset\\dish_ids\\splits\\rgb_test_ids.txt"

# Load data
df = pd.read_csv(cleaned_csv)

# Load train/test IDs
with open(train_ids_path, 'r') as f:
    train_ids = set(line.strip() for line in f)

with open(test_ids_path, 'r') as f:
    test_ids = set(line.strip() for line in f)

# Filter the DataFrame
df_train = df[df["dish_id"].isin(train_ids)]
df_test = df[df["dish_id"].isin(test_ids)]

# Save split files
df_train.to_csv("D:\\Thesis\\segmentation_train.csv", index=False)
df_test.to_csv("D:\\Thesis\\segmentation_test.csv", index=False)

print("✅ Train and test splits created and saved.")
