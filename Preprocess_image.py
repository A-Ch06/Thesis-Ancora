from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

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

# Load the model
model_path = "D:\\Thesis\\Model"
model = tf.saved_model.load(model_path)
infer = model.signatures["default"]

def process_image(image_path):
    image = Image.open(image_path).resize((513, 513))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  

    # Convert to Tensor
    input_tensor = tf.constant(image, dtype=tf.float32)

    # Run inference
    output = infer(images=input_tensor)

    # Extract predictions
    segmentation_mask = output["food_group_segmenter:semantic_predictions"].numpy()[0]

    return segmentation_mask

def display_results(image_path, segmentation_mask):

    # Create the color map (using tab20 and extending for more classes)
    tab20 = plt.get_cmap('tab20')
    tab20b = plt.get_cmap('tab20b')
    tab20c = plt.get_cmap('tab20c')
    combined_colors = list(tab20.colors) + list(tab20b.colors) + list(tab20c.colors)
    cmap = mcolors.ListedColormap(combined_colors[:len(FOOD_CLASSES)])

    # Normalize the segmentation mask to [0, 1] for color mapping
    norm = mcolors.Normalize(vmin=0, vmax=len(FOOD_CLASSES) - 1)

    # Map the segmentation mask to colors using the tab20 colormap
    colored_mask = cmap(norm(segmentation_mask))

    # Convert to uint8 for display
    colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)


    plt.figure(figsize=(10, 8))
    plt.imshow(colored_mask)
    plt.title("Segmentation Output with Color-Coding")
    plt.axis("off")

    # Create a plot for the labels and their respective colors
    plt.figure(figsize=(5, len(FOOD_CLASSES)/4)) 
    for i, food_class in enumerate(FOOD_CLASSES):
        plt.plot([0, 1], [i, i], color=cmap(i), lw=6) 
        plt.text(1.05, i, food_class, ha='left', va='center', fontsize=5, color='black')  

    plt.title("Food Class Labels with Corresponding Colors")
    plt.axis("off")
    plt.tight_layout(pad=2.0) 

    plt.show()

def process_images_in_folder(folder_path):
    """
    Process and display results for all images in a folder.
    """
    # Get all image files from the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process each image in the folder
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_path}...")
        segmentation_mask = process_image(image_path)
        display_results(image_path, segmentation_mask)

folder_path = "D:\\Thesis\\Food_Images" 
process_images_in_folder(folder_path)





#Using the map calculate the area of the pixels, and relate it to the area of pixels found in the nutrition5k and aproximate the calories, carbs, and nutrients based on that and also the grams.