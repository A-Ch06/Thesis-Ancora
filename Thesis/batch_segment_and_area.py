import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# ====== CONFIGURATION ======
MODEL_PATH = "D:\\Thesis\\Model"
ROOT_DIR = "D:\\Thesis\\nutrition5k_data\\nutrition5k_dataset\\imagery\\realsense_overhead"
SEGMENTED_SAVE_DIR = "D:\\Thesis\\segmented_masks"
CSV_OUTPUT_PATH = "D:\\Thesis\\outputs\\segmentation_areas.csv"

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

# ====== CREATE OUTPUT FOLDERS ======
os.makedirs(SEGMENTED_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)

# ====== LOAD MODEL ======
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

# ====== PROCESS IMAGE ======
def process_image(image_path):
    image = Image.open(image_path).resize((513, 513))
    image = np.array(image) / 255.0
    input_tensor = tf.constant(image[None, ...], dtype=tf.float32)
    output = infer(images=input_tensor)
    return output["food_group_segmenter:semantic_predictions"].numpy()[0]

# ====== SAVE MASK IMAGE ======
def save_segmented_image(mask, save_path):
    colored_mask = cmap(norm(mask))[:, :, :3]
    colored_mask = (colored_mask * 255).astype(np.uint8)
    Image.fromarray(colored_mask).save(save_path)

# ====== COMPUTE PIXEL AREA VECTOR ======
def compute_class_areas(mask):
    areas = np.zeros(len(FOOD_CLASSES), dtype=int)
    for i in range(len(FOOD_CLASSES)):
        areas[i] = np.sum(mask == i)
    return areas

# ====== MAIN PIPELINE ======
def process_all_dishes(root_dir):
    results = []
    for dish_folder in os.listdir(root_dir):
        dish_path = os.path.join(root_dir, dish_folder)
        image_path = os.path.join(dish_path, "rgb.png")
        if not os.path.exists(image_path):
            continue

        print(f"Processing: {dish_folder}")
        try:
            mask = process_image(image_path)
        except Exception as e:
            print(f"Failed to process {dish_folder}: {e}")
            continue

        # Save segmented mask
        segmented_filename = f"{dish_folder}_segmented.png"
        save_path = os.path.join(SEGMENTED_SAVE_DIR, segmented_filename)
        save_segmented_image(mask, save_path)

        # Compute pixel areas
        area_vector = compute_class_areas(mask)
        results.append([dish_folder] + area_vector.tolist())

    # Save all results to CSV
    df = pd.DataFrame(results, columns=["dish_id"] + FOOD_CLASSES)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\n✅ Done! CSV saved to: {CSV_OUTPUT_PATH}")
    print(f"✅ Segmented images saved to: {SEGMENTED_SAVE_DIR}")

process_all_dishes(ROOT_DIR)
