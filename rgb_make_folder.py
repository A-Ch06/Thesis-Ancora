import os
import shutil

# ====== CONFIGURATION ======
SOURCE_ROOT_DIR = r"D:/Thesis/nutrition5k_data/nutrition5k_dataset/imagery/realsense_overhead"
DEST_DIR = r"D:/Thesis/rgb_photos"

# ====== CREATE DESTINATION FOLDER ======
os.makedirs(DEST_DIR, exist_ok=True)

# ====== MAIN PROCESS ======
def copy_rgb_images():
    count = 0
    for dish_id in os.listdir(SOURCE_ROOT_DIR):
        dish_path = os.path.join(SOURCE_ROOT_DIR, dish_id)
        rgb_path = os.path.join(dish_path, "rgb.png")

        if os.path.isfile(rgb_path):
            dest_filename = f"{dish_id}_rgb.png"
            dest_path = os.path.join(DEST_DIR, dest_filename)
            shutil.copy(rgb_path, dest_path)
            count += 1
            print(f"Copied: {dish_id} → {dest_filename}")
        else:
            print(f"Skipped: {dish_id} (no rgb.png found)")

    print(f"\nDone! {count} images copied to: {DEST_DIR}")

# ====== RUN ======
if __name__ == "__main__":
    copy_rgb_images()
