import pandas as pd
import ast
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import os

from frontend.regression.utils.utils_preprocessing import correct_term, parse_range, unit_categories, unit_conversions, standardize_measurements, standardize_temperature

ps = PorterStemmer()


def remove_stop_word(recipes):
    nltk.download('stopwords')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    additional = {
        "minutes", "easiest", "ever", "aww", "i", "can", "t", "believe", "it", "s", "stole", "the", "idea", "from","mirj", "andrea", " s ", "andreas",
        "viestad", "andes", "andersen", "an", "ana", "amy", "2 ww points", "on demand", "anelia", "amazing",
        "ashley", "ashton", "amazing", "make", "house", "smell", "malcolm", "amazingly", "killer", "perfect",
        "addictive", "leave", "u", "licking", "ur", "finger", "clean", "th", "recipe", "special", "time", "favorite",
        "aunt", "jane", "soft", "and", "moist", "licking", "famous", "non fruitcake", "true", "later",
        "nonbeliever", "believer", "comfort", "ultimate", "lover", "love", "easy", "ugly", "cc", "uncle", "bill", "tyler",
        "unbelievably", "unbelievable", "healthy", "fat", "free", "un", "melt", "mouth", "ummmmm", "umm", "ummmy", "nummy", "ummmm", "unattended",
        "unbaked", "ultra", "ultimately", "yummy", "rich", "quick", "rachael", "ray", "fail", "party", "florence",
        "fast", "light", "low", "carb", "snack", "wedding", "anniversary", "anne", "marie", "annemarie", "annette", "funicello", "syms",
        "byrn", "mike", "willan", "summer", "autumn", "winter", "spring", "burrel", "anna", "tres", "sweet", "uber",
        "homemade", "ann","best","j", "anite", "anitas", "anman", "angie", "angry", "simple", "difficult", "andy", "andrew", "ancient", "still", "another", "best", "go",
        "grant", "grandma", "amusement", "park", "instruction", "kitchen", "test", "ww", "almost", "empty", "dressing", "instant", "like", "le", "virtually",
        "home", "made", "guilt", "guilty", "delicious", "parfait", "forgotten", "forget", "forevermama", "diet", "can", "real", "former",
        "miss", "fabulous", "forever", "authentic", "fortnum", "mason", "kid", "foolproof", "football", "season", "diabetic",
        "two", "small", "one", "three", "four", "five", "thanksgiving", "dream", "foothill", "paula", "deen", "food", "processor", "safari", "processor",
        "traditional", "forbidden", "flavorful", "grandmag", "grandmama", "grandmaman", "grandma", "grandmom", "lena", "alicia", "alisa", "alice", "ali", "bit", "different",
        "eat", "family", "global", "gourmet", "yam", "yam", "emotional", "balance", "tonight", "feel", "cooking", "got", "birthday", "air", "way", "mr", "never", "weep", "half",
        "anything", "pour", "put", "fork", "say", "stove", "top", "thought", "prize", "winning", "add", "ad", "good", "better", "da", "style", "even", "bran", "fake", "fire", "beautiful"
        "l", "game", "day", "hate", "world", "minute", "type", "starbucks", "biggest", "dressed", "summertime", "elmer", "johnny", "depp", "c", "p", "h", "clove", "er", "star", "week",
        "affair", "elegant", "student", "z", "whole", "lotta", "w", "z", "b", "aaron", "craze", "a", "abc", "absolute", "absolut", "absolutely", "perfection", "delightful", "lazy", "morning",
        "abuelo", "abuelito", "abuelita", "abuela", "acadia", "accidental", "adam", "little", "interest", "addicting", "addie", "adele", "adelaide", "adi", "adie", "adriana",
        "adult", "affordable", "alison", "holst", "purpose", "allegheny", "allegedly", "original", "allergic", "ex", "allergy", "allergen", "allen", "poorman", "backyard",
        "alton", "brown", "whatever", "anthony", "anytime", "april", "fool", "ya", "fooled", "sandra", "lee", "edna", "emma", "emy", "evy", "eva", 'evelyn', "fannie", "fanny", "flo", "gladys", "helen", "grace", "ira", "irma",
        "isse", "jean", "janet", "jenny", "juju", "judy", "kathy", "kathi", "kellie", "kelly", "laura", "lee", "kay", "kathleen", "laura", "lee", "lesley", "lil", "linda", "liz", "lois", "louisse",
        "mag", 'martguerite', "margie", "marge", "maggie", "martha", "marylin", "marion", "mary", "marthy", "melody", "michel", "meda", "millie", "muriel", "myrna", "nelda", "nancy", "paulie", "phillis", "rae", "rebecca",
        "rose", "sadie", "sarah", "sara", "sue", "susan", "teresa", "theresa", "auntie", "em", "barbara", "barb", "irene", "lolo", "lori", "lu", "maebelle",
        "aunty", "aussie", "aurora", "austin", "l", "q"
        
        }
    stop_words.update(additional) 
    cleaned_recipes = []
        
    for recipe in recipes:

        recipe = recipe.lower()
        recipe = re.sub(r'[^a-z\s]', '', recipe)
        
        recipe_words = recipe.split()
        
        # Lemmatize first
        recipe_words = [lemmatizer.lemmatize(word) for word in recipe_words]
        
        # Then remove stopwords
        recipe_words = [word for word in recipe_words if word not in stop_words]
        
        cleaned_recipe = " ".join(recipe_words)
        cleaned_recipes.append(cleaned_recipe)
    
    return cleaned_recipes

    
def normalize_columns(data):
    data["description"] = data["description"].apply(lambda x: x.lower())

    columns = ["tags", "steps", "ingredients"]

    for c in columns:
        data[c] = data[c].apply(lambda x : [s.lower() for s in x])

    return data


def standardize_units(text):
    """
    Main function to standardize all units in text.
    This refactored approach processes different unit types in separate passes.
    """
    # Step 1: First convert temperatures (to avoid conflicts with other patterns)
    result = standardize_temperature(text)
    
    # Step 2: Handle measurement standardizations (inches, dimensions)
    result = standardize_measurements(result)
    
    # Step 3: Now process the remaining units
    words = result.lower().split()
    result_words = []
    i = 0
    
    while i < len(words):
        word = words[i]
        next_word = words[i+1] if i + 1 < len(words) else ""
        next2_word = words[i+2] if i + 2 < len(words) else ""
        next3_word = words[i+3] if i + 3 < len(words) else ""

        # Handle fractions like "1 / 2 inch"
        if (
            i + 2 < len(words)
            and re.match(r"^\d+(\.\d+)?$", word)
            and words[i+1] == "/"
            and re.match(r"^\d+(\.\d+)?$", words[i+2])
        ):
            numerator = float(word)
            denominator = float(words[i+2])
            fraction_value = numerator / denominator
            
            # Check if there's a unit after the fraction
            if i + 3 < len(words):
                corrected_unit = correct_term(words[i+3])
                if corrected_unit in unit_categories:
                    category = unit_categories[corrected_unit]
                    converted = fraction_value * unit_conversions.get(corrected_unit, 1)
                    result_words.append(f"{converted} {category}")
                    i += 4  # Move past the fraction and the unit
                    continue
            
            # If no unit or unrecognized unit, just keep the fraction as a decimal
            result_words.append(str(fraction_value))
            i += 3
            continue

        # Handle "2-3 kg"
        value = parse_range(word, next_word)
        if isinstance(value, float) and next_word:
            corrected_unit = correct_term(next_word)
            if corrected_unit in unit_categories:
                category = unit_categories[corrected_unit]
                converted = value * unit_conversions.get(corrected_unit, 1)
                result_words.append(f"{converted} {category}")
                i += 2
                continue

        # Handle "2 to 3 kg"
        if value == "to" and next2_word.replace('.', '', 1).isdigit() and next3_word:
            average = (float(word) + float(next2_word)) / 2
            corrected_unit = correct_term(next3_word)
            if corrected_unit in unit_categories:
                category = unit_categories[corrected_unit]
                converted = average * unit_conversions.get(corrected_unit, 1)
                result_words.append(f"{converted} {category}")
                i += 4
                continue

        # Handle regular value + unit
        if re.match(r"^\d+(\.\d+)?$", word) and next_word:
            corrected_unit = correct_term(next_word)
            if corrected_unit in unit_categories:
                category = unit_categories[corrected_unit]
                converted = float(word) * unit_conversions.get(corrected_unit, 1)
                result_words.append(f"{converted} {category}")
                i += 2
                continue

        # Default: stem and append
        result_words.append(ps.stem(correct_term(word)))
        i += 1

    return " ".join(result_words)
    
    
def expand_nutrition_column(data):
    data['nutrition'] = data['nutrition'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    if data['nutrition'].apply(lambda x: isinstance(x, list)).all():
        data[['calories', 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']] = pd.DataFrame(data['nutrition'].to_list(), index=data.index)
    
        data.drop(columns=['nutrition'], inplace=True)
        
    return data

def process_data(save=False):
    print(os.getcwd())
    data  = pd.read_csv('data/RAW_recipes.csv')
    data.set_index('id', inplace=True)
    columns = ["tags", "steps", "ingredients", "nutrition"]

    for i in columns:
        data[i] = data[i].apply(ast.literal_eval)

    data.drop(columns=["contributor_id", "submitted"], inplace=True, errors="ignore")
    data.dropna(subset=["name"], inplace=True)
    data = data[data['minutes'] < 300]

    data['name'] = remove_stop_word(data['name'])

    data.dropna(subset=['name', 'description'], inplace=True)
    data.reset_index(inplace=True)
    data = normalize_columns(data)
    
    data["steps_strings"] = data["steps"].apply(lambda x : ' '.join(x))
    data["steps_string_standardize"] = data["steps_strings"].apply(standardize_units)

    data["ingredients_text"] = data["ingredients"].apply(lambda x: ' '.join(x))
    data["ingredients_text"] = data["ingredients"].astype(str)

    data["tags"] = data["tags"].apply(
        lambda tags: [tag for tag in tags if not any(keyword in tag.lower() for keyword in ["minute", "time", "hours", "preparation"])]
    )
    data["tags_text"] = data["tags"].apply(lambda x: ' '.join(x))
    data["tags_text"] = data["tags"].astype(str)
    
    
    data = expand_nutrition_column(data)
    data.drop(columns=['nutrition_score'], inplace=True, errors='ignore')
    data.drop(columns=['ingredients', 'steps', 'steps_strings', 'tags'], inplace=True, errors='ignore')
    data = data[data['steps_string_standardize'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    data.reset_index(inplace=True)
    
    if (save):
        os.makedirs('data/processed', exist_ok=True)
        data.to_csv('data/processed/preprocessed_recipe.csv', index=False)
    return data
    