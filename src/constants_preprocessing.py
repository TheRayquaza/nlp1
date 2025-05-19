from nltk.corpus import stopwords

STANDARD_VOLUME = "liter"
STANDARD_TIME = "minute"
STANDARD_WEIGHT = "gram"
STANDARD_TEMP = "celsius"

# 1. Your tag → cuisine mapping
TAG2CLASS = {
    # North America – United States
    "american": "North America – United States",
    "north-american": "North America – United States",
    "northeastern-united-states": "North America – United States",
    "californian": "North America – United States",
    "native-american": "North America – United States",
    "pennsylvania-dutch": "North America – United States",
    "hawaiian": "North America – United States",

    # North America – Canada
    "canadian": "North America – Canada",
    "british-columbian": "North America – Canada",
    "quebec": "North America – Canada",

    # Central America & Caribbean
    "mexican": "Central America & Caribbean",
    "costa-rican": "Central America & Caribbean",
    "guatemalan": "Central America & Caribbean",
    "caribbean": "Central America & Caribbean",
    "cuban": "Central America & Caribbean",
    "puerto-rican": "Central America & Caribbean",
    "creole": "Central America & Caribbean",

    # South America
    "argentine": "South America",
    "brazilian": "South America",
    "peruvian": "South America",
    "chilean": "South America",
    "colombian": "South America",
    "venezuelan": "South America",
    "ecuadorean": "South America",

    # Europe – Western
    "french": "Europe – Western",
    "english": "Europe – Western",
    "scottish": "Europe – Western",
    "irish": "Europe – Western",
    "welsh": "Europe – Western",
    "dutch": "Europe – Western",
    "belgian": "Europe – Western",
    "austrian": "Europe – Western",
    "german": "Europe – Western",
    "italian": "Europe – Western",
    "portuguese": "Europe – Western",

    # Europe – Northern
    "swedish": "Europe – Northern",
    "norwegian": "Europe – Northern",
    "finnish": "Europe – Northern",
    "icelandic": "Europe – Northern",
    "danish": "Europe – Northern",

    # Europe – Eastern/Central
    "russian": "Europe – Eastern",
    "hungarian": "Europe – Eastern",
    "czech": "Europe – Eastern",
    "georgian": "Europe – Eastern",

    # Middle East & North Africa (MENA)
    "middle-eastern": "Middle East & North Africa",
    "turkish": "Middle East & North Africa",
    "lebanese": "Middle East & North Africa",
    "iranian-persian": "Middle East & North Africa",
    "iraqi": "Middle East & North Africa",
    "palestinian": "Middle East & North Africa",
    "saudi-arabian": "Middle East & North Africa",
    "egyptian": "Middle East & North Africa",
    "moroccan": "Middle East & North Africa",
    "libyan": "Middle East & North Africa",

    # Sub-Saharan Africa
    "south-african": "Sub-Saharan Africa",
    "ethiopian": "Sub-Saharan Africa",
    "nigerian": "Sub-Saharan Africa",
    "angolan": "Sub-Saharan Africa",
    "sudanese": "Sub-Saharan Africa",

    # Asia – East
    "chinese": "Asia – East",
    "beijing": "Asia – East",
    "chinese-new-year": "Asia – East",
    "japanese": "Asia – East",
    "korean": "Asia – East",

    # Asia – Southeast
    "vietnamese": "Asia – Southeast",
    "indonesian": "Asia – Southeast",
    "malaysian": "Asia – Southeast",
    "cambodian": "Asia – Southeast",
    "laotian": "Asia – Southeast",

    # Asia – South
    "pakistani": "Asia – South",
    "nepalese": "Asia – South",

    # Asia – Central & Inner
    "mongolian": "Asia – Central",

    # Oceania & Pacific
    "polynesian": "Oceania & Pacific",

    # Jewish Diaspora
    "jewish-ashkenazi": "Jewish Diaspora",
    "jewish-sephardi": "Jewish Diaspora",

    # Catch-all / generic
    "asian": "Asia – General",
    "european": "Europe – General",
}

# 2. Build a prioritized list of tags, from most specific → most generic
PRIORITIZED_TAGS = [
    # Europe – Western
    "french", "english", "scottish", "irish", "welsh",
    "dutch", "belgian", "austrian", "german", "italian",
    # North America – United States
    "american", "north-american", "northeastern-united-states",
    "californian", "native-american", "pennsylvania-dutch",
    "hawaiian",
    # North America – Canada
    "canadian", "british-columbian", "quebec",
    # Central America & Caribbean
    "mexican", "costa-rican", "guatemalan",
    "caribbean", "cuban", "puerto-rican", "creole",
    # South America
    "argentine", "brazilian", "peruvian",
    "chilean", "colombian", "venezuelan", "ecuadorean",
    # Europe – Northern
    "swedish", "norwegian", "finnish", "icelandic", "danish",
    # Europe – Eastern/Central
    "russian", "hungarian", "czech", "georgian",
    # Europe – Southern
    "portuguese",
    # Middle East & North Africa (MENA)
    "middle-eastern", "turkish", "lebanese",
    "iranian-persian", "iraqi", "palestinian",
    "saudi-arabian", "egyptian", "moroccan", "libyan",
    # Sub-Saharan Africa
    "south-african", "ethiopian", "nigerian",
    "angolan", "sudanese",
    # Asia – East
    "chinese", "beijing", "chinese-new-year",
    "japanese", "korean",
    # Asia – Southeast
    "vietnamese", "indonesian", "malaysian",
    "cambodian", "laotian",
    # Asia – South
    "pakistani", "nepalese",
    # Asia – Central & Inner
    "mongolian",
    # Oceania & Pacific
    "polynesian",
    # Jewish Diaspora
    "jewish-ashkenazi", "jewish-sephardi",
    # Finally, the generic catch-alls
    "asian", "european",
]

# Unit conversion factors to standardized units
UNIT_CONVERSIONS = {
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
UNIT_CATEGORIES = {
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

COMMON_UNITS = list(UNIT_CATEGORIES.keys())

TYPO_CORRECTIONS = {
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

UNITS = ["liter", "minute", "gram", "hours", "cup", "quart", "mL", "second", "week", 
         "pound", "ounce", "kg", "oz", "g", "°c", "°f", "fahrenheit", "degre", "celsius"]

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS_ADDITIONAL = {
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
STOP_WORDS.update(STOP_WORDS_ADDITIONAL) 