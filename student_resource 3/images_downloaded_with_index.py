import pandas as pd
import os
from src.utils import download_image

# Load the test data
test_data = pd.read_csv('./dataset/test.csv')

# Filter the test data for height entity
height_data = test_data[test_data['entity_name'] == 'height']

# Create a folder for storing height images
image_folder = './test_images/height'
os.makedirs(image_folder, exist_ok=True)

# Modified function to download images with index as filenames
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

# Re-download the images with the index as the filename
download_images_with_index(height_data, image_folder)
print("Images re-downloaded with index as filenames.")