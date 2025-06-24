from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
MODEL_PATH = r"D:\Thesis\Model"
print("Loading model...")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["default"]

# Path to test image
image_path = r"D:\Thesis\augmentation\contrast_low\dish_1558637932_rgb.png"

# Force RGB and resize
try:
    image = Image.open(image_path).convert("RGB").resize((513, 513))
    image_np = np.array(image) / 255.0

    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError(f"Unexpected shape: {image_np.shape}")

    input_tensor = tf.constant(image_np[None, ...], dtype=tf.float32)
    output = infer(images=input_tensor)
    mask = output["food_group_segmenter:semantic_predictions"].numpy()[0]

    print("Segmentation succeeded")
except Exception as e:
    print(f"Failed with error:\n{e}")
