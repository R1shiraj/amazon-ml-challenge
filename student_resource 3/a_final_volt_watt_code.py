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

# Filter the test data for voltage and wattage entities
voltage_data = test_data[test_data['entity_name'] == 'voltage']
wattage_data = test_data[test_data['entity_name'] == 'wattage']

# Image folder paths (Assuming images are already downloaded in respective folders)
voltage_image_folder = './test_images/voltage'
wattage_image_folder = './test_images/wattage'

# Function to extract text from an image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    text = pytesseract.image_to_string(eroded_image)
    return text

# Convert the parsed value to the desired units for voltage and wattage
def convert_to_standard_units(value, unit, entity_name):
    unit = unit.lower()
    value = float(value)
    
    if entity_name == "voltage":
        if unit in {"v", "volt", "volts"}:
            return value, "volt"
        elif unit in {"kv", "kilovolt", "kilovolts"}:
            return value , "kilovolt"
        elif unit in {"mv", "millivolt", "millivolts"}:
            return value , "millivolt"
    elif entity_name == "wattage":
        if unit in {"w", "watt", "watts"}:
            return value, "watt"
        elif unit in {"kw", "kilowatt", "kilowatts"}:
            return value , "kilowatt"
    
    return None, None  # If no valid unit is found

# Function to standardize units
def standardize_unit(text):
    unit_mapping = {
        "v": "volt",
        "volt": "volt",
        "volts": "volt",
        "kv": "kilovolt",
        "kilovolt": "kilovolt",
        "kilovolts": "kilovolt",
        "mv": "millivolt",
        "millivolt": "millivolt",
        "millivolts": "millivolt",
        "w": "watt",
        "watt": "watt",
        "watts": "watt",
        "kw": "kilowatt",
        "kilowatt": "kilowatt",
        "kilowatts": "kilowatt"
    }
    
    # Replace all units with standard units
    for short_unit, standard_unit in unit_mapping.items():
        text = re.sub(rf"\b{short_unit}\b", standard_unit, text.lower())
    
    return text

# Function to extract measurements
def extract_measurements(text):
    measurements = []
    # Regex to find numeric values with units
    pattern = r"(\d+\.?\d*)\s*(volt|volts|v|kilovolt|kilovolts|kv|millivolt|millivolts|mv|watt|watts|w|kilowatt|kilowatts|kw)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    for match in matches:
        value, unit = match
        value = float(value)
        measurements.append((value, unit))
    
    return measurements

# Function to predict voltage or wattage
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
        filtered_measurements = [convert_to_standard_units(value, unit, entity_name) for value, unit in measurements]
        
        if len(filtered_measurements) == 0:
            predictions.append({"index": row['index'], "prediction": ""})
            continue

        prediction = filtered_measurements[0]

        # Create prediction string
        prediction_str = f"{prediction[0]} {prediction[1]}" if prediction[0] else ""
        # Append the result with only index and prediction
        predictions.append({"index": row['index'], "prediction": prediction_str})
    
    # Create a dataframe for the predictions
    predictions_df = pd.DataFrame(predictions)
    
    return predictions_df

# Predict voltage
voltage_predictions = predict_entity(voltage_data, 'voltage', voltage_image_folder)

# Predict wattage
wattage_predictions = predict_entity(wattage_data, 'wattage', wattage_image_folder)

# Merge voltage and wattage predictions into a final CSV
final_predictions = pd.concat([voltage_predictions, wattage_predictions], ignore_index=True)

# Save the final CSV with only 'index' and 'prediction' columns
final_predictions[['index', 'prediction']].to_csv('./dataset/voltage_and_wattage_predictions.csv', index=False)
print("Final predictions for all images saved to voltage_and_wattage_predictions.csv")
