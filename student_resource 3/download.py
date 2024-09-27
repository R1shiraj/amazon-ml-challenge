import pandas as pd
import os
from src.utils import download_images  # Assuming the function is imported from utils

# Load the dataset
train_data = pd.read_csv('./dataset/train.csv')

# Get the unique entity names
unique_entities = train_data['entity_name'].unique()

# Base folder for storing the downloaded images
base_folder = './test_images'

# Create the base folder if it doesn't exist
if not os.path.exists(base_folder):
    os.makedirs(base_folder)

# Loop through each unique entity and download all images for each
x = ['height', 'width', 'depth']
for entity in x:
    # Create a folder for each unique entity
    entity_folder = os.path.join(base_folder, entity)
    
    if not os.path.exists(entity_folder):
        os.makedirs(entity_folder)
    
    # Get the first 5 image links corresponding to the current entity
    entity_images = train_data[train_data['entity_name'] == entity]['image_link'].head(25000).tolist()
    
    # Download the images into the respective folder
    download_images(image_links=entity_images, download_folder=entity_folder, allow_multiprocessing=False)

print("Images downloaded for each entity in their respective folders.")
