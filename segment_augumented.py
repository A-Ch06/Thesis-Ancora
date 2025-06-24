import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import gc

# ====== CONFIGURATION ======
MODEL_PATH = r"D:\Thesis\Model"
AUGMENTED_ROOT_DIR = r"D:\Thesis\augmentation"
SEGMENTED_SAVE_DIR = r"D:\Thesis\segmented_augmentation"
CSV_OUTPUT_DIR = r"D:\Thesis\outputs\segmentation_areas"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Could not set memory growth:", e)

# ====== FOOD CLASS LABELS ======

FOOD_CLASSES = [
    "background", 
    "vegetables | leafy_greens", 
    "vegetables | stem_vegetables", 
    "vegetables | non-starchy_roots", 
    "vegetables | other", 
    "fruits", 
    "protein | meat", 
    "protein | poultry", 
    "protein | seafood", 
    "protein | eggs", 
    "protein | beans/nuts", 
    "starches/grains | baked_goods", 
    "starches/grains | rice/grains/cereals", 
    "starches/grains | noodles/pasta", 
    "starches/grains | starchy_vegetables", 
    "starches/grains | other", 
    "soups/stews", 
    "herbs/spices", 
    "dairy", 
    "snacks", 
    "sweets/desserts", 
    "beverages", 
    "fats/oils/sauces", 
    "food_containers", 
    "dining_tools", 
    "other_food"
]
# ====== INITIAL SETUP ======
os.makedirs(SEGMENTED_SAVE_DIR, exist_ok=True)
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

print("Loading model...")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["default"]

# ====== COLOR MAP ======
tab20 = plt.get_cmap('tab20')
tab20b = plt.get_cmap('tab20b')
tab20c = plt.get_cmap('tab20c')
combined_colors = list(tab20.colors) + list(tab20b.colors) + list(tab20c.colors)
cmap = mcolors.ListedColormap(combined_colors[:len(FOOD_CLASSES)])
norm = mcolors.Normalize(vmin=0, vmax=len(FOOD_CLASSES) - 1)

# ====== HELPER FUNCTIONS ======
def process_image(image_path):
    image = Image.open(image_path).resize((513, 513))
    image = np.array(image) / 255.0
    input_tensor = tf.constant(image[None, ...], dtype=tf.float32)
    output = infer(images=input_tensor)
    return output["food_group_segmenter:semantic_predictions"].numpy()[0]

def save_segmented_image(mask, save_path):
    colored_mask = cmap(norm(mask))[:, :, :3]
    colored_mask = (colored_mask * 255).astype(np.uint8)
    Image.fromarray(colored_mask).save(save_path)

def compute_class_areas(mask):
    areas = np.zeros(len(FOOD_CLASSES), dtype=int)
    for i in range(len(FOOD_CLASSES)):
        areas[i] = np.sum(mask == i)
    return areas

# ====== MAIN FUNCTION ======
def process_augmented_images(root_dir):
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing folder: {folder}")
        results = []

        for filename in os.listdir(folder_path):
            if not filename.endswith(".png"):
                continue

            image_path = os.path.join(folder_path, filename)
            dish_id = filename.split("_")[0]

            try:
                mask = process_image(image_path)
            except Exception as e:
                print(f"Failed: {filename} — {e}")
                continue

            # Save segmented image
            seg_folder = os.path.join(SEGMENTED_SAVE_DIR, folder)
            os.makedirs(seg_folder, exist_ok=True)
            seg_filename = filename.replace(".png", "_segmented.png")
            seg_path = os.path.join(seg_folder, seg_filename)
            save_segmented_image(mask, seg_path)

            # Save area vector
            area_vector = compute_class_areas(mask)
            results.append([filename, dish_id] + area_vector.tolist())

        # Save CSV
        if results:
            csv_path = os.path.join(CSV_OUTPUT_DIR, f"{folder}.csv")
            df = pd.DataFrame(results, columns=["filename", "dish_id"] + FOOD_CLASSES)
            df.to_csv(csv_path, index=False)
            print(f"CSV saved: {csv_path}")

        # ==== CLEANUP after this folder ====
        tf.keras.backend.clear_session()
        gc.collect()
        print("TensorFlow session cleared.")

# ====== RUN ======
if __name__ == "__main__":
    process_augmented_images(AUGMENTED_ROOT_DIR)
