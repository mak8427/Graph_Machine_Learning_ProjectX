import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Define the dataset URL and destination folder
dataset_slug = "davidemattioli/pose-dataset"
destination_folder = "./pose-dataset"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Download the dataset
print(f"Downloading dataset: {dataset_slug}")
api.dataset_download_files(dataset_slug, path=destination_folder, unzip=True)

print(f"Dataset downloaded and unzipped to: {destination_folder}")

