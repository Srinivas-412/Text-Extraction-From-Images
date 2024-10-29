import os
import random
import pandas as pd
import cv2
import requests
import numpy as np
import easyocr
import re

def predictor(image_link, category_id, entity_name):
    '''
    Call your model/approach here
    '''
    # Fetch the image from the URL
    response = requests.get(image_link)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        print("Error loading image!")
        return ""

    # Use EasyOCR to read text from the image
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img, paragraph=True)

    # Extract text from OCR results
    text_list = [detection[1] for detection in result]

    # Extract and correct entities from the OCR text
    extracted_data = extract_entities_and_correct(text_list)

    # Filter extracted data based on entity_name
    filtered_data = filter_extracted_data_by_entity(extracted_data, entity_name, entity_unit_map)

    # Select the best value from the filtered data
    best_value = select_best_value(filtered_data)

    return best_value if best_value else ""

def extract_entities_and_correct(text_list):
    text = ' '.join(text_list)

    # Apply custom OCR correction
    text = correct_ocr_mistakes(text)

    # Correct common misspellings or variations
    corrections = {
        'grams': 'gram', 'g': 'gram', 'gm': 'gram', 'milligrams': 'milligram', 'mg': 'milligram',
        'millilitres': 'millilitre', 'ml': 'millilitre', 'litres': 'litre', 'l': 'litre',
        'centimetres': 'centimetre', 'cm': 'centimetre', 'metres': 'metre', 'm': 'metre',
        'millimetres': 'millimetre', 'mm': 'millimetre', 'kilograms': 'kilogram', 'kg': 'kilogram',
        'pounds': 'pound', 'lbs': 'pound', 'lb': 'pound', 'ounces': 'ounce', 'oz': 'ounce',
        'gallons': 'gallon', 'pints': 'pint', 'quarts': 'quart', 'cups': 'cup', 'tons': 'ton',
        'millivolts': 'millivolt', 'volts': 'volt', 'kilovolts': 'kilovolt', 'watts': 'watt', 'kilowatts': 'kilowatt',
        'CM':'centimetre', 'ibs':'pound'
    }

    # Replace misspellings in the text
    for wrong, right in corrections.items():
        text = re.sub(rf'\b{wrong}\b', right, text, flags=re.IGNORECASE)

    # Find matches with the pattern
    pattern = r'(\d+(\.\d+)?)\s*(mg|milligram|g|gram|kg|kilogram|ounce|pound|ton|cm|centimetre|mm|millimetre|m|metre|lb|lbs|watt|watts|kilowatt|kW|ml|millilitre|V|volt|w|W|inch|in|ft|feet|%)|(\d+)\s*(lbs|pounds|foot|feet|inch|in|meter|m|centimeter|cm|millimeter|mm|watt|watts|w|W)'
    matches = re.findall(pattern, text, re.IGNORECASE)

    # Extract and correct data
    extracted_data = []
    for match in matches:
        value, unit = match[0], match[2].lower()  # Normalize units to lowercase
        corrected_unit = corrections.get(unit, unit)  # Correct the unit if it’s misspelled

        if corrected_unit in allowed_units:
            try:
                value = float(value)  # Convert to float for consistent formatting
                extracted_data.append({'value': value, 'unit': corrected_unit})
            except ValueError:
                continue

    return extracted_data

def filter_extracted_data_by_entity(extracted_data, entity_name, entity_unit_map):
    valid_units = entity_unit_map.get(entity_name, set())  # Get valid units for the entity_name
    filtered_data = [item for item in extracted_data if item['unit'] in valid_units]
    return filtered_data if filtered_data else None

def select_best_value(filtered_data):
    if not filtered_data:
        return None
    best_entry = max(filtered_data, key=lambda x: x['value'])  # Assuming max value is the best
    formatted_value = "{:.2f}".format(best_entry['value'])
    return f"{formatted_value} {best_entry['unit']}"

# Custom OCR correction function
def correct_ocr_mistakes(text):
    corrections = {
        'O': '0',  # Letter O to digit 0
        'CM': 'cm',  # Normalize to lowercase
        'IBS': 'lbs',  # Correct misreadings for pounds
        'W': 'watt',  # Normalize watt
        'Im': 'lm',  # Lumens abbreviation
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

if __name__ == "__main__":
    DATASET_FOLDER = '../dataset/'
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)

    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
