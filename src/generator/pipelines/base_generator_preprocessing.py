import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from typing import List
import string
import pickle
import warnings
import ast
from collections import Counter, defaultdict
import pandas as pd
import ast
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from typing import List, Dict, Set
import string
import pickle
import warnings
import ast
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

class BaseGeneratorPreprocessing:
    def __init__(self,
                 ingredients_col='ingredients_text',
                 steps_col='steps_string_standardize',
                 drop_uncommon=None):
        self.ingredients_col = ingredients_col
        self.steps_col = steps_col
        self.stemmer = LancasterStemmer()
        self.drop_uncommon = drop_uncommon

    def _extract_ingredients(self, row) -> List[str]:
        if pd.isna(row[self.ingredients_col]):
            return []

        ingredients = ast.literal_eval(row[self.ingredients_col])
        return [self.stemmer.stem(ing) for ing in ingredients if ing]

    def _extract_steps(self, row) -> List[str]:
        if pd.isna(row[self.steps_col]):
            return []
    
        steps = row[self.steps_col].split(',')
        steps = [step.strip().lower() for step in steps]
    
        cleaned_stems = []
        for step in steps:
            cleaned = re.sub(r'[^a-zA-Z0-9 .]', '', step)
    
            if not cleaned or cleaned.isdigit():
                continue
    
            stemmed = self.stemmer.stem(cleaned)
            cleaned_stems.append(stemmed)
    
        return cleaned_stems

    def fit(self, X: pd.DataFrame, y=None):
        missing_cols = [col for col in [self.ingredients_col, self.steps_col] if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in the DataFrame")
        return self

    def transform(self, X: pd.DataFrame) -> Dict[str, List[List[str]]]:
        X = X.dropna(subset=[self.ingredients_col, self.steps_col])
        ingredients_lists = []
        steps_lists = []
        ingredients_counter = defaultdict(int)

        for _, row in X.iterrows():
            ingredients = self._extract_ingredients(row)
            steps = self._extract_steps(row)
            ingredients_lists.append(ingredients)
            steps_lists.append(steps)
            for ing in ingredients:
                ingredients_counter[ing] += 1

        if self.drop_uncommon is not None:
            uncommon_ingredients = {
                ing for ing, count in ingredients_counter.items()
                if count < self.drop_uncommon
            }
    
            filtered_ingredients_lists = []
            filtered_steps_lists = []
            for ingredients, steps in zip(ingredients_lists, steps_lists):
                if not any(ing in uncommon_ingredients for ing in ingredients):
                    filtered_ingredients_lists.append(ingredients)
                    filtered_steps_lists.append(steps)
    
            ingredients_lists = filtered_ingredients_lists
            steps_lists = filtered_steps_lists
        return ingredients_lists, steps_lists

    def fit_transform(self, X: pd.DataFrame, y=None) -> Dict[str, List[List[str]]]:
        return self.fit(X).transform(X)