import pandas as pd
import os
import re
import cv2
import pytesseract
import logging

# Path to Tesseract-OCR executable (adjust if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# logging.basicConfig(filename='prediction_log.txt', level=logging.INFO)

# Load the test data
test_data = pd.read_csv('./dataset/test.csv')

# Filter the test data for width and depth entities
width_data = test_data[test_data['entity_name'] == 'width'].head(20)  # Process only 20 images for width
depth_data = test_data[test_data['entity_name'] == 'depth'].head(20)  # Process only 20 images for depth

# Image folder paths (Assuming images are already downloaded in respective folders)
width_image_folder = './test_images/width'
depth_image_folder = './test_images/depth'

# Allowed units map
entity_unit_map = {
    "width": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "depth": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
}

# Conversion factors to standardize all units to centimetres
unit_conversion_factors = {
    "centimetre": 1,
    "millimetre": 0.1,
    "metre": 100,
    "inch": 2.54,
    "foot": 30.48,
    "yard": 91.44
}

# Function to extract text from an image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image)
    return text

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

# Function to convert measurements to a standardized unit (centimetre)
def convert_to_standard_unit(value, unit):
    if unit in unit_conversion_factors:
        return value * unit_conversion_factors[unit]
    return value

# Function to predict width or depth
def predict_entity(entity_data, entity_name, image_folder):
    predictions = []
    
    for _, row in entity_data.iterrows():
        image_name = f"{row['index']}.jpg"
        image_path = os.path.join(image_folder, image_name)
        
        # Ensure the image exists
        if not os.path.exists(image_path):
            predictions.append({"index": row['index'], "prediction": ""})
            # logging.info(f"Index: {row['index']}, Image not found.")
            continue
        
        # Extract text from the image
        extracted_text = extract_text_from_image(image_path)
        
        # Standardize units in the extracted text
        standardized_text = standardize_unit(extracted_text)
        
        # Extract measurements from the standardized text
        measurements = extract_measurements(standardized_text)

        if len(measurements) == 0:
            predictions.append({"index": row['index'], "prediction": ""})
            logging.info(f"Index: {row['index']}, No measurements found.")
            continue

        # Log the extracted measurements for debugging
        logging.info(f"Index: {row['index']}, Extracted Text: {extracted_text}, Measurements: {measurements}")
        
        # Convert measurements to the standardized unit
        converted_measurements = [(convert_to_standard_unit(value, unit), 'centimetre') for value, unit in measurements]
        print("Converted measurements: ", converted_measurements)

        # Filter measurements for the allowed units
        filtered_measurements = [(value, unit) for value, unit in converted_measurements if unit in entity_unit_map[entity_name]]
        print("Filtered measurements: ", filtered_measurements)

        if len(filtered_measurements) == 0:
            predictions.append({"index": row['index'], "prediction": ""})
            logging.info(f"Index: {row['index']}, No valid measurements found after filtering.")
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
        print(prediction)
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
final_predictions[['index', 'prediction']].to_csv('./dataset/final_predictions_10_images.csv', index=False)
print("Final predictions for 10 images saved to final_predictions_10_images.csv")
