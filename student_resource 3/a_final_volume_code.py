import pandas as pd
import os
import re
import cv2
import pytesseract
import logging

# Set up logging to file
logging.basicConfig(filename='extracted_texts_and_measurements.txt',
                    filemode='w',  # Use 'a' to append logs to the file
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Path to Tesseract-OCR executable (adjust if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the test data
test_data = pd.read_csv('./dataset/test.csv')

# Filter the test data for "item_volume" entity
item_volume_data = test_data[test_data['entity_name'] == 'item_volume']

# Image folder paths (Assuming images are already downloaded in the respective folder)
item_volume_image_folder = './test_images/item_volume'

# Function to extract text from an image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.mlLcfzoi'  # Adjust for volume units
    text = pytesseract.image_to_string(binary_image, config=config)
    logging.info(f'Extracted text from {image_path}: {text}')
    return text

# Convert the parsed value to the desired units (for volume)
def convert_to_standard_units(value, unit):
    unit = unit.lower()
    value = float(value)
    
    # Conversion logic based on the target unit
    if unit in ["ml", "millilitre", "millilitres", "mls", "ML", "mL", "mLs"]:
        return value, "millilitre"
    elif unit in ["l", "litre", "litres", "ls", "L", "Ls"]:
        return value, "litre"
    elif unit in ["f", "o", "FL", "OZ"]:
        return value, "fluid ounce"
    elif unit in ["cubic inch", "cubic inches", "cu in"]:
        return value, "cubic inch"
    elif unit in ["gallon", "gallons"]:
        return value, "gallon"
    elif unit in ["pint", "pints"]:
        return value, "pint"
    elif unit in ["quart", "quarts"]:
        return value, "quart"
    elif unit in ["cup", "cups"]:
        return value, "cup"
    elif unit in ["microlitre", "microlitres"]:
        return value, "microlitre"
    elif unit in ["centilitre", "centilitres"]:
        return value, "centilitre"
    elif unit in ["imperial gallon", "imperial gallons"]:
        return value, "imperial gallon"
    elif unit in ["cubic foot", "cubic feet"]:
        return value, "cubic foot"
    else:
        return None, None

# Function to standardize units
def standardize_unit(text):
    unit_mapping = {
        "ml": "millilitre",
        "l": "litre",
        "litre": "litre",
        "fl oz": "fluid ounce",
        "oz": "fluid ounce",
        "gallon": "gallon",
        "pint": "pint",
        "quart": "quart",
        "cup": "cup",
        "microlitre": "microlitre",
        "centilitre": "centilitre",
        "imperial gallon": "imperial gallon",
        "cubic inch": "cubic inch",
        "cubic foot": "cubic foot",
    }

    # Replace all units with standard units
    for short_unit, standard_unit in unit_mapping.items():
        text = re.sub(rf"\b{short_unit}\b", standard_unit, text.lower())
    
    logging.info(f'Standardized text: {text}')
    return text

# Function to extract measurements
def extract_measurements(text):
    measurements = []
    # Regex to find numeric values with units
    pattern = r"(\d+\.?\d*)\s*(millilitre|litre|fluid ounce|cubic inch|gallon|pint|quart|cup|microlitre|centilitre|cubic foot|fl oz)"
    matches = re.findall(pattern, text, re.IGNORECASE)

    for match in matches:
        value, unit = match
        value = float(value)
        measurements.append((value, unit))
    
    logging.info(f'Extracted measurements: {measurements}')
    return measurements

# Function to predict item volume
def predict_entity(entity_data, entity_name, image_folder):
    predictions = []

    for _, row in entity_data.iterrows():
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
        print("Measurements: ", measurements)

        # Convert measurements to the standardized unit
        filtered_measurements = [(convert_to_standard_units(value, unit)) for value, unit in measurements]

        if len(filtered_measurements) == 0:
            predictions.append({"index": row['index'], "prediction": ""})
            continue

        # Select the largest volume value
        prediction = max(filtered_measurements, key=lambda x: x[0])

        # Create prediction string
        prediction_str = f"{prediction[0]} {prediction[1]}" if prediction[0] else ""
        predictions.append({"index": row['index'], "prediction": prediction_str})

        # Log the final prediction
        logging.info(f'Prediction for image {image_name}: {prediction_str}')

    # Create a dataframe for the predictions
    predictions_df = pd.DataFrame(predictions)

    return predictions_df

# Predict item volume
item_volume_predictions = predict_entity(item_volume_data, 'item_volume', item_volume_image_folder)

# Save the final CSV with only 'index' and 'prediction' columns
item_volume_predictions[['index', 'prediction']].to_csv('./dataset/item_volume_predictions.csv', index=False)

# Load the CSV file into a DataFrame
df = pd.read_csv('./dataset/item_volume_predictions.csv')

# Specify the column name you're interested in
column_name = 'prediction'

# Count the number of non-null entries in the column
total_entries = df[column_name].count()

print(f'Total number of entries in column "{column_name}": {total_entries}')

print("Final predictions for item volume saved to item_volume_predictions.csv")
