import os

# Path to your overhead dataset directory
dataset_path = "D:\\Thesis\\nutrition5k_data\\nutrition5k_dataset\\imagery\\realsense_overhead"

# Output file for missing IDs
output_file = "D:\\Thesis\\missing_rgb_dish_ids.txt"

# Collect missing dish IDs
missing_dish_ids = []

# Loop through all folders
for dish_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, dish_folder)
    rgb_path = os.path.join(folder_path, "rgb.png")
    
    if not os.path.exists(rgb_path):
        missing_dish_ids.append(dish_folder)

# Save the missing dish IDs
with open(output_file, "w") as f:
    for dish_id in missing_dish_ids:
        f.write(f"{dish_id}\n")

print(f"✅ Found {len(missing_dish_ids)} missing rgb.png images.")
print(f"Saved list to: {output_file}")
