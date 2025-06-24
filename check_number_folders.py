import os

# Path to image directory
dataset_path = "D:\\Thesis\\nutrition5k_data\\nutrition5k_dataset\\imagery\\realsense_overhead"

# List all items in the directory
dish_folders = [name for name in os.listdir(dataset_path) 
                if os.path.isdir(os.path.join(dataset_path, name)) and name.startswith("dish_")]

# Count unique folders
unique_folder_count = len(set(dish_folders))

print(f"Found {unique_folder_count} unique dish folders in: {dataset_path}")
