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

# Filter the test data for maximum weight and item weight entities
max_weight_data = test_data[test_data['entity_name'] == 'maximum_weight_recommendation']  # Process all images for maximum weight
item_weight_data = test_data[test_data['entity_name'] == 'item_weight']  # Process all images for item weight

# Image folder paths (Assuming images are already downloaded in respective folders)
max_weight_image_folder = './test_images/maximum_weight_recommendation'
item_weight_image_folder = './test_images/item_weight'

# Function to extract text from an image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    # eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.kgGLbm'  # Whitelist only numbers and units
    text = pytesseract.image_to_string(binary_image, config=config)

    return text

# Convert the parsed value to the desired units (for weight)
def convert_to_standard_units(value, unit):
    unit = unit.lower()
    value = float(value)
    
    # Conversion logic based on the target unit
    if unit in ["kg", "kilograms", "kilogram", "kilos", "kgs", "kgs", "kilo"]:
        return value, "kilogram"
    elif unit in ["g", "gs", "gram", "grams", "gm", "gms"]:
        return value , "gram"  
    elif unit in ["mg", "milligram", "mgs", "milligrams"]:
        return value , "milligram"  
    elif unit in ["lb", "pound", "lbs", "pounds", "LBS", "LBs", "LB"]:
        return value , "pound"  
    elif unit in ["ton", "tons"]:
        return value , "ton"
    elif unit in ["oz", "ozs", "ooz", "ounce", "ounces"]:
        return value , "ounce"
    elif unit in ["mcg", "mcgs", "micrograms", "microgram"]:
        return value , "microgram"
    else:
        return None, None  # If no valid unit is found

# Function to standardize units
def standardize_unit(text):
    unit_mapping = {
        "kg": "kilogram",
        "g": "gram",
        "mg": "milligram",
        "lb": "pound",
        "lbs": "pound",
        "ton": "ton",
        "tons": "ton",
        "kilo": "kilogram",
        "kilogram": "kilogram",
        "grams": "gram",
        "milligram": "milligram",
        "pounds": "pound",
    }
    
    # Replace all units with standard units
    for short_unit, standard_unit in unit_mapping.items():
        text = re.sub(rf"\b{short_unit}\b", standard_unit, text.lower())
    
    return text

# Function to extract measurements
def extract_measurements(text):
    measurements = []
    # Regex to find numeric values with units
    pattern = r"(\d+\.?\d*)\s*(kilogram|gram|milligram|pound|kg|g|mg|lb|lbs|ton|tons)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    for match in matches:
        value, unit = match
        value = float(value)
        measurements.append((value, unit))
    
    return measurements

# Function to predict maximum weight or item weight
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

        if len(filtered_measurements) == 0:
            predictions.append({"index": row['index'], "prediction": ""})
            continue

        # Selection logic based on entity type (maximum weight or item weight)
        if entity_name in ["maximum_weight_recommendation", "item_weight"]:
            # For both maximum weight and item weight, select the largest value
            prediction = max(filtered_measurements, key=lambda x: x[0])

        # Create prediction string
        prediction_str = f"{prediction[0]} {prediction[1]}" if prediction[0] else ""
        # Append the result with only index and prediction
        predictions.append({"index": row['index'], "prediction": prediction_str})
    
    # Create a dataframe for the predictions
    predictions_df = pd.DataFrame(predictions)
    
    return predictions_df

# Predict maximum weight
max_weight_predictions = predict_entity(max_weight_data, 'maximum_weight_recommendation', max_weight_image_folder)

# Predict item weight
item_weight_predictions = predict_entity(item_weight_data, 'item_weight', item_weight_image_folder)

# Merge maximum weight and item weight predictions into a final CSV
final_predictions = pd.concat([max_weight_predictions, item_weight_predictions], ignore_index=True)

# Save the final CSV with only 'index' and 'prediction' columns
final_predictions[['index', 'prediction']].to_csv('./dataset/weights_predictions.csv', index=False)

print("Final predictions for all images saved to weights_predictions.csv")
