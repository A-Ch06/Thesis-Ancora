import tarfile
import os

# Define paths
tar_path = "D:\Thesis\mobile-food-segmenter-v1-tensorflow1-seefood-segmenter-mobile-food-segmenter-v1-v1.tar.gz"
extract_path = "D:\Thesis\Model"

# Extract the .tar.gz file
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(extract_path)

print("Model extracted to:", extract_path)