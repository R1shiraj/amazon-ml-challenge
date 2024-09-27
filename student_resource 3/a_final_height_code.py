import pytesseract
import cv2
import os
import pandas as pd
import re
import numpy as np

# Path to Tesseract-OCR executable (adjust if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Preprocess the image for better OCR detection
def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Image at {image_path} could not be read.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray

# Extract text from an image using Tesseract OCR
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Identify vertical lines in the image (to help identify height)
def detect_vertical_lines(image):
    # Use Canny edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Use HoughLinesP to detect vertical lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=5)

    if lines is None:
        print("No lines detected.")
        return []

    # Filter out only vertical lines (approximate by looking at line angles)
    vertical_lines = [line for line in lines if abs(line[0][0] - line[0][2]) < 10]  # Allow some tolerance

    return vertical_lines

# Extract text from a region near the center of the line
def extract_text_near_line(image, line):
    x1, y1, x2, y2 = line[0]
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Define a region around the center of the line
    margin = 50
    height, width = image.shape
    region = image[max(center_y - margin, 0):min(center_y + margin, height),
                   max(center_x - margin, 0):min(center_x + margin, width)]

    # Extract text from the region
    text = pytesseract.image_to_string(region)
    return text

# Regex to capture number and unit (cm, mm, inch, etc.)
def parse_dimensions(text):
    matches = re.findall(r'(\d+\.?\d*)\s?(cm|mm|inch|foot|yard|metre)', text, re.IGNORECASE)
    return matches

# Convert the parsed value to the desired units (for height)
def convert_to_standard_units(value, unit):
    unit = unit.lower()
    value = float(value)
    
    # Conversion logic based on the target unit
    if unit == "cm":
        return value, "centimetre"
    elif unit == "mm":
        return value , "millimetre"
    elif unit == "inch" or unit == "in" or unit == '"':
        return value, "inch"
    elif unit == "foot":
        return value , "foot"
    elif unit == "yard":
        return value , "yard"
    elif unit == "m":
        return value , "metre"
    else:
        return None, None  # If no valid unit is found

# Process the images and extract height information
def process_images(df, image_folder, output_file):
    results = []
    
    for idx, row in df.iterrows():
        image_path = os.path.join(image_folder, f"{row['index']}.jpg")
        
        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist, skipping...")
            continue
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image is None:
            continue
        
        # Detect vertical lines (to help identify height)
        vertical_lines = detect_vertical_lines(preprocessed_image)
        
        # Extract text from the preprocessed image
        text = extract_text_from_image(preprocessed_image)
        print(f"Extracted Text for Image {image_path}: {text}")  # Print extracted text
        
        if text == "":
            continue  # Skip if no text was extracted
        
        dimensions = parse_dimensions(text)
        
        # If dimensions are found, extract the text near the vertical lines
        height = ""
        if dimensions:
            for value, unit in dimensions:
                # Check if any vertical lines are detected
                if vertical_lines:
                    for line in vertical_lines:
                        line_text = extract_text_near_line(preprocessed_image, line)
                        print(f"Text near line: {line_text}")
                        # Refine the logic to select the height based on the extracted text near lines
                        converted_value, converted_unit = convert_to_standard_units(value, unit)
                        if converted_value is not None:
                            height = f"{converted_value} {converted_unit}"
                            break
                    if height:
                        break
        
        # If no vertical lines were detected, return any measurement found as text
        if not height and dimensions:
            height = f"{dimensions[0][0]} {dimensions[0][1]}"
        
        # Append result with index and height
        results.append({
            'index': row['index'],
            'prediction': height
        })
    
    # Save results to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)

# Example usage: process all "height" images and save the output
test_data = pd.read_csv('./dataset/test.csv')  # Path to your test.csv file

# Filter only for the "height" entity
height_data = test_data[test_data['entity_name'] == 'height']

# Process the height images only
process_images(height_data, './test_images/height', './height_predictions.csv')
