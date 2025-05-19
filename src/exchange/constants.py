from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
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
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.update(additional)
