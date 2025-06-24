import tarfile

file_path = "D:\Thesis\\nutrition5k_dataset.tar"

with tarfile.open(file_path, "r:gz") as tar:
    tar.extractall("nutrition5k_data")  # Extract into this folder