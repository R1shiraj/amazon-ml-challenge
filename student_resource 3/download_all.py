import pandas as pd
import os
from src.utils import download_image

# Load the test data
test_data = pd.read_csv('./dataset/test.csv')

# List of entity names and their corresponding folder names
entities = [
    'item_weight', 
    'depth', 
    'width', 
    'voltage', 
    'wattage', 
    'item_volume', 
    'maximum_weight_recommendation'
]

# Function to download images with index as filenames
def download_images_with_index(df, image_folder):
    # Ensure the image folder exists
    os.makedirs(image_folder, exist_ok=True)
    
    # Iterate over the dataframe and download each image
    for _, row in df.iterrows():
        image_url = row['image_link']
        image_name = f"{row['index']}.jpg"  # Save with index as the filename
        save_path = os.path.join(image_folder, image_name)  # Full path to save the image

        # Download the image with the new filename (index-based)
        download_image(image_link=image_url, save_folder=image_folder, save_path=save_path)

# Iterate over each entity and download images into respective folders
for entity in entities:
    # Filter the test data for the current entity
    entity_data = test_data[test_data['entity_name'] == entity]
    
    # Create a folder for storing the entity images
    image_folder = f'./test_images/{entity}'
    os.makedirs(image_folder, exist_ok=True)
    
    print(f"Downloading images for {entity}...")
    
    # Re-download the images with the index as the filename
    download_images_with_index(entity_data, image_folder)

print("All images downloaded for all entities.")
