import re
import pandas as pd
from nltk.stem import PorterStemmer
from rapidfuzz import process, fuzz


STANDARD_VOLUME = "liter"
STANDARD_TIME = "minute"
STANDARD_WEIGHT = "gram"
STANDARD_TEMP = "celsius"

# Unit conversion factors to standardized units
unit_conversions = {
    # Volume conversions to liter
    "cup": 0.24,         # 1 cup = 0.24 liters
    "quart": 0.95,       # 1 quart = 0.95 liters
    "mL": 0.001,         # 1 mL = 0.001 liters
    
    # Time conversions to minutes
    "second": 1/60,      # 1 second = 1/60 minutes
    "hours": 60,         # 1 hour = 60 minutes
    "week": 10080,       # 1 week = 10080 minutes
    
    # Weight conversions to grams
    "pound": 453.59,     # 1 pound = 453.59 grams
    "ounce": 28.35,      # 1 ounce = 28.35 grams
    "kg": 1000,          # 1 kg = 1000 grams
    "g": 1,              # 1 g = 1 gram

    # Inch to centimeter conversion
    "inch": 2.54,        # 1 inch = 2.54 cm
    "inches": 2.54,      # 1 inch = 2.54 cm
}

# Mapping of units to their standard category
unit_categories = {
    # Volume units
    "cup": STANDARD_VOLUME,
    "quart": STANDARD_VOLUME,
    "mL": STANDARD_VOLUME,
    "liter": STANDARD_VOLUME,
    
    # Time units
    "second": STANDARD_TIME,
    "minutes": STANDARD_TIME,
    "hours": STANDARD_TIME,
    "week": STANDARD_TIME,
    
    # Weight units
    "pound": STANDARD_WEIGHT,
    "ounce": STANDARD_WEIGHT,
    "kg": STANDARD_WEIGHT,
    "g": STANDARD_WEIGHT,
    "gram": STANDARD_WEIGHT,

    # Measurement units
    "inch": "cm",
    "inches": "cm",
    
    # Temperature units
    "°c": STANDARD_TEMP,
    "°f": STANDARD_TEMP,
    "celsius": STANDARD_TEMP,
    "fahrenheit": STANDARD_TEMP,
}

common_units = list(unit_categories.keys())

typo_corrections = {
    "gram": "g",
    "gm": "g",
    "lb": "pound",
    "oz": "ounce",
    "kilogram": "kg",
    "centimetr": "cm",
    "centimet": "cm",
    "mm": "mL",
    "millimet": "mL",
    "millilit": "mL",
    "centigrad": "°c",
    "litr": "liter",
    "cupof": "cup of",
    "talbespoon": "tablespoon",
    "tablespon": "tablespoon",
    "tablesppoon": "tablespoon",
    "tblpss": "tablespoon",
    "tbso": "tablespoon",
    "tbspn": "tablespoon",
    "tbslp": "tablespoon",
    "tsbp": "tablespoon",
    "tlbsp": "tablespoon",
    "tablestoon": "tablespoon",
    "tablepoon": "tablespoon",
    "teasppon": "teaspoon",
    "teapsoon": "teaspoon",
    "teaspon": "teaspoon",
    "teaspoom": "teaspoon",
    "cu": "cup",
    
    # Temperature corrections
    "f": "°f",
    "fahrenheit": "°f",
    "c": "°c",
    "celsius": "°c",
    "degrees f": "°f",
    "degrees c": "°c",
    "degree f": "°f",
    "degree c": "°c",

    # Times
    "min" : "minutes",
    "minuet": "minutes",
    "minutesthen": "minutes",
    "minutesor": "minutes",
    "minutesyour": "minutes",
    "minuet": "minutes",
    "miniut": "minutes",
    "mimut": "minutes",
    "mionut": "minutes",
    "mintur": "minutes",
    "mkinut": "minutes",
    "mminut": "minutes",
    "munut": "minutes",
    "minuest": "minutes",
    "minunet": "minutes",
    "mintes": "minutes",
    "mutes": "minutes",
    "mutesr": "minutes",
    "minutesr": "minutes",
    "minuteslong": "minutes long",
    "minutesbrush": "minutes brush",
    "minnut": "minutes",
    "minuteuntil": "minutes until",
    "minutesm": "minutes",
    "nminut": "minutes",
    "minit": "minutes",
    "minutu": "minutes",
    "mihnut": "minutes",
    "mintut": "minutes",
    "minutr": "minutes",
    "ninut": "minutes",
    "minutew": "minutes",
    "minutess": "minutes",
    "minutesssssssss": "minutes",
    "minuteswil": "minutes will",
    "seccond": "second",
    "secong": "second",
    "seceond": "second",
    "housr": "hours",
    "houir": "hours",
    "hoursin": "hours",
    "hoursovernight": "hours overnight",
    "secon": "second",
    "seccond": "second",
    "secong": "second",
    "seceond": "second",
    "wk": "week",
    "hr": "hours",
    "b": "lb",
    "z": "oz",
    "″": "inch",
    '"': "inch" 
}

def fahrenheit_to_celsius(f):
    """Convert Fahrenheit to Celsius and round to 2 decimal places"""
    return round((f - 32) * 5.0 / 9.0, 2)
def standardize_temperature(text):
    """
    Process temperature mentions in text, converting Fahrenheit to Celsius,
    but preserving temperatures that are already in Celsius.
    Also avoids confusing time durations with temperatures.
    """
    # First handle explicit Celsius (just standardize format without changing the value)
    # Handle this BEFORE Fahrenheit to avoid double conversion
    c_pattern = r'(\d+(?:\.\d+)?)\s*(?:degrees?\s*c|°c|\^c|celsius)'
    result_text = re.sub(c_pattern, 
                     lambda m: f"{float(m.group(1))} {STANDARD_TEMP}", 
                     text, 
                     flags=re.IGNORECASE)
    
    # Also handle direct C notation
    direct_c_pattern = r'(\d+(?:\.\d+)?)([°]?c\b)'
    result_text = re.sub(direct_c_pattern, 
                     lambda m: f"{float(m.group(1))} {STANDARD_TEMP}", 
                     result_text, 
                     flags=re.IGNORECASE)
    
    # Now handle explicit Fahrenheit temperatures, which we definitely want to convert
    # Handle explicit Fahrenheit with degree symbol or text
    f_pattern = r'(\d+(?:\.\d+)?)\s*(?:degrees?\s*f|°f|\^f|fahrenheit)'
    
    def process_temp(match):
        temp_value = float(match.group(1))
        return f"{fahrenheit_to_celsius(temp_value)} {STANDARD_TEMP}"
    
    result_text = re.sub(f_pattern, process_temp, result_text, flags=re.IGNORECASE)
    
    # Handle direct F notation
    direct_f_pattern = r'(\d+(?:\.\d+)?)([°]?f\b)'
    result_text = re.sub(direct_f_pattern, process_temp, result_text, flags=re.IGNORECASE)
    
    # For standalone "350 degrees" without explicit F/C, we should be careful
    # We'll only convert if we can strongly infer it's Fahrenheit (like high cooking temps)
    pattern_standalone_degrees = r'(\d+(?:\.\d+)?)\s*(?:degrees?|°)(?!\s*[fc]|\s*fahrenheit|\s*celsius)'
    
    def process_ambiguous_temp(match):
        value = float(match.group(1))
        # Only convert if very likely Fahrenheit (high cooking temps > 200)
        if value > 200:
            return f"{fahrenheit_to_celsius(value)} {STANDARD_TEMP}"
        # For temperatures that could be either F or C (e.g., 100 degrees),
        # preserve the original text to avoid incorrect conversions
        return match.group(0)
    
    result_text = re.sub(pattern_standalone_degrees, process_ambiguous_temp, result_text, flags=re.IGNORECASE)
    
    # Then handle cooking context temperatures (preheat, heat, bake, etc.) CAREFULLY
    # We need to avoid matching phrases like "bake 15 minutes"
    # This pattern specifically looks for temperatures WITHOUT explicit Celsius indication
    cooking_temp_pattern = r'((?:preheat|heat|oven|temperature|temp)(?:\s+to)?)\s+(\d+(?:\.\d+)?)(?:\s*(?:degrees?|°)|\b)(?!\s*(?:minute|min|hour|sec|day|week))(?!\s*(?:c|celsius|°c))'
    
    def process_cooking_temp(match):
        context = match.group(1)
        value = float(match.group(2))
        # For cooking temperatures:
        # - Values below 100: Could be C, don't convert
        # - Values 100-200: Ambiguous zone, examine more carefully
        # - Values above 200: Very likely F, convert to C
        if value > 200:
            return f"{context} {fahrenheit_to_celsius(value)} {STANDARD_TEMP}"
        # For ambiguous or likely Celsius values, preserve original
        return match.group(0)
    
    result_text = re.sub(cooking_temp_pattern, process_cooking_temp, result_text, flags=re.IGNORECASE)
    
    # Special case for "bake at X" or "cook at X" where X is a temperature
    # Only match cases NOT explicitly marked as Celsius
    bake_at_pattern = r'((?:bake|cook)(?:\s+at)?)\s+(\d+(?:\.\d+)?)(?:\s*(?:degrees?|°)|\b)(?!\s*(?:minute|min|hour|sec|day|week))(?!\s*(?:c|celsius|°c))'
    
    def process_bake_temp(match):
        context = match.group(1)
        value = float(match.group(2))
        # Similar logic as cooking temperatures
        # Only convert values that are very likely to be Fahrenheit
        if value > 200:
            return f"{context} {fahrenheit_to_celsius(value)} {STANDARD_TEMP}"
        # For ambiguous temperatures or temperatures likely in Celsius already, preserve original
        return match.group(0)
    
    result_text = re.sub(bake_at_pattern, process_bake_temp, result_text, flags=re.IGNORECASE)
    
    return result_text

def standardize_measurements(text):
    """
    Handle measurement-specific standardizations, especially for dimensions like 9x5"
    """
    # Keep the dimension format (NxM) but convert each number from inches to cm
    dimension_pattern = r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:"|″|inch(?:es)?)?'
    result = re.sub(dimension_pattern, 
                    lambda m: f"{float(m.group(1)) * 2.54}x{float(m.group(2)) * 2.54} cm", 
                    text)
    
    # Handle single inch measurements
    inch_pattern = r'(\d+(?:\.\d+)?)(?:"|″|inch(?:es)?)'
    result = re.sub(inch_pattern, 
                    lambda m: f"{float(m.group(1)) * 2.54} cm", 
                    result)
    
    return result

def correct_term(word):
    """Apply fuzzy matching to correct typos in unit terms"""
    # If it is a number return the word
    if not any(c.isalpha() for c in word):
        return word

    # Check if in mapping
    if word in typo_corrections:
        return typo_corrections[word]
      
    # Fuzzy matching
    match, score, _ = process.extractOne(word, common_units, scorer=fuzz.ratio)
    if score > 80:
        return match
    return word

def parse_range(word, next_word=None):
    """
    Detects numeric ranges like "2-3" or "2 to 3" and returns their mean as a float.
    E.g. => 2-3 kgs becomes 2.5 kgs
    """
    if re.match(r"^\d+(\.\d+)?-\d+(\.\d+)?$", word):  # "2-3"
        start, end = map(float, word.split("-"))
        return (start + end) / 2
    if next_word and word.isdigit() and next_word == "to":
        return "to" 
    return None
