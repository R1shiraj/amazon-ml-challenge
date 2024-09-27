import pandas as pd
import os
import re
import cv2
import pytesseract
from tqdm import tqdm  # Import tqdm for progress bar

# Path to Tesseract-OCR executable (adjust if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the test data
test_data = pd.read_csv('./dataset/test.csv')

# Filter the test data for width and depth entities
width_data = test_data[test_data['entity_name'] == 'width']  # Process all images for width
depth_data = test_data[test_data['entity_name'] == 'depth']  # Process all images for depth

# Image folder paths (Assuming images are already downloaded in respective folders)
width_image_folder = './test_images/width'
depth_image_folder = './test_images/depth'

# Function to extract text from an image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Convert the parsed value to the desired units (for height)
def convert_to_standard_units(value, unit):
    unit = unit.lower()
    value = float(value)
    
    # Conversion logic based on the target unit
    if unit == "cm" or unit == "centimeter" or unit == "centimetre":
        return value, "centimetre"
    elif unit == "mm" or unit == "millimeter" or unit == "millimetre":
        return value , "millimetre"
    elif unit == "inch" or unit == "in" or unit == '"' or unit == "inches":
        return value, "inch"
    elif unit == "foot" or unit == "ft" or unit == "'":
        return value , "foot"
    elif unit == "yard" or unit == "yd":
        return value , "yard"
    elif unit == "m" or unit == "meter" or unit == "metre":
        return value , "metre"
    else:
        return None, None  # If no valid unit is found

# Function to standardize units
def standardize_unit(text):
    unit_mapping = {
        "cm": "centimetre",
        "mm": "millimetre",
        "m": "metre",
        '"': "inch",
        "inches": "inch",
        "in": "inch",  # Handle 'in' abbreviation
        "ft": "foot",
        "yd": "yard",
        "'": "foot",  # Handle ' symbol as foot
        "centimeter": "centimetre",
        "millimeter": "millimetre",
        "meter": "metre"
    }
    
    # Replace all units with standard units
    for short_unit, standard_unit in unit_mapping.items():
        text = re.sub(rf"\b{short_unit}\b", standard_unit, text.lower())
    
    return text

# Function to extract measurements
def extract_measurements(text):
    measurements = []
    # Regex to find numeric values with units
    pattern = r"(\d+\.?\d*)\s*(centimetre|millimetre|metre|inch|in|feet|foot|yard|cm|mm|m|ft|yd|inches|\"|\')"
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    for match in matches:
        value, unit = match
        value = float(value)
        measurements.append((value, unit))
    
    return measurements

# Function to predict width or depth
def predict_entity(entity_data, entity_name, image_folder):
    predictions = []
    
    # Add a progress bar
    for _, row in tqdm(entity_data.iterrows(), total=entity_data.shape[0], desc=f'Processing {entity_name}'):
        image_name = f"{row['index']}.jpg"
        image_path = os.path.join(image_folder, image_name)
        
        # Ensure the image exists
        if not os.path.exists(image_path):
            predictions.append({"index": row['index'], "prediction": ""})
            continue
        
        # Extract text from the image
        extracted_text = extract_text_from_image(image_path)
        
        # Standardize units in the extracted text
        standardized_text = standardize_unit(extracted_text)
        
        # Extract measurements from the standardized text
        measurements = extract_measurements(standardized_text)

        # Convert measurements to the standardized unit
        filtered_measurements = [(convert_to_standard_units(value, unit)) for value, unit in measurements]

        # Filter out None values
        filtered_measurements = [(value, unit) for value, unit in filtered_measurements if value is not None and unit is not None]


        if len(filtered_measurements) == 0:
            predictions.append({"index": row['index'], "prediction": ""})
            continue

        # Selection logic based on entity type (width or depth)
        if entity_name == "depth":
            # For depth, select the smallest value
            prediction = min(filtered_measurements, key=lambda x: x[0])
        else:  # entity_name == "width"
            # For width, return the second measurement if possible, otherwise the first one
            if len(filtered_measurements) >= 2:
                prediction = filtered_measurements[1]  # Second measurement
            else:
                prediction = filtered_measurements[0]  # First (and only) measurement

        # Create prediction string
        prediction_str = f"{prediction[0]} {prediction[1]}" if prediction[0] else ""
        # Append the result with only index and prediction
        predictions.append({"index": row['index'], "prediction": prediction_str})
    
    # Create a dataframe for the predictions
    predictions_df = pd.DataFrame(predictions)
    
    return predictions_df

# Predict width
width_predictions = predict_entity(width_data, 'width', width_image_folder)

# Predict depth
depth_predictions = predict_entity(depth_data, 'depth', depth_image_folder)

# Merge width and depth predictions into a final CSV
final_predictions = pd.concat([width_predictions, depth_predictions], ignore_index=True)

# Save the final CSV with only 'index' and 'prediction' columns
final_predictions[['index', 'prediction']].to_csv('./dataset/depth_and_width_predictions.csv', index=False)
print("Final predictions for all images saved to depth_and_width_predictions.csv")
