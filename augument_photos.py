import os
import cv2
import numpy as np

# Paths
input_folder = "D:/Thesis/rgb_photos"
output_root = "augmentation"

LOW_RES_SIZE = (432, 432)   # 85% of 512
MEDIUM_RES_SIZE = (384, 384)  # 75% of 512

#low and medium refers to the intensity at which we apply the augumentation
augmentations = {
    "resolution_low": lambda img: cv2.resize(cv2.resize(img, LOW_RES_SIZE), img.shape[:2][::-1]),
    "resolution_medium": lambda img: cv2.resize(cv2.resize(img, MEDIUM_RES_SIZE), img.shape[:2][::-1]),

    "brightness_low": lambda img: cv2.convertScaleAbs(img, alpha=1.0, beta=-15),
    "brightness_medium": lambda img: cv2.convertScaleAbs(img, alpha=1.0, beta=-30),

    "contrast_low": lambda img: cv2.convertScaleAbs(img, alpha=0.8, beta=0),
    "contrast_medium": lambda img: cv2.convertScaleAbs(img, alpha=0.5, beta=0),

    "noise_low": lambda img: np.clip(img + np.random.normal(0, 10, img.shape).astype(np.int16), 0, 255).astype(np.uint8),
    "noise_medium": lambda img: np.clip(img + np.random.normal(0, 25, img.shape).astype(np.int16), 0, 255).astype(np.uint8),
}

# Process each augmentation
for aug_name, aug_func in augmentations.items():
    aug_folder = os.path.join(output_root, aug_name)
    os.makedirs(aug_folder, exist_ok=True)

    for img_name in os.listdir(input_folder):
        if img_name.endswith(".png"):
            img_path = os.path.join(input_folder, img_name)
            img = cv2.imread(img_path)

            augmented_img = aug_func(img)
            save_path = os.path.join(aug_folder, img_name)
            cv2.imwrite(save_path, augmented_img)

print("Augmentation completed.")
