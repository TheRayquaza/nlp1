import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from typing import List, Dict, Set, Tuple
import string
import pickle
import warnings
import ast
from collections import Counter, defaultdict
from transformers import AutoTokenizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class AutotokenizerGeneratorPreprocessing:
    def __init__(self,
                 ingredients_col='ingredients_text',
                 steps_col='steps_string_standardize',
                 drop_uncommon=None):
        self.ingredients_col = ingredients_col
        self.steps_col = steps_col
        self.stemmer = LancasterStemmer()
        self.drop_uncommon = drop_uncommon
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

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
    
    def _collect_corpus(self, X: pd.DataFrame) -> List[str]:
        corpus = []
        for _, row in X.iterrows():
            ingredients = self._extract_ingredients(row)
            steps = self._extract_steps(row)
            
            if ingredients:
                corpus.extend(ingredients)
            if steps:
                corpus.extend(steps)
        return corpus

    def tokenize(self, text: str) -> List[str]:
        pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits = [[l for l in word] for word in pre_tokenized_text]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])
    
    def tokenize_list(self, text_list: List[str]) -> List[List[str]]:
        return [self.tokenize(text) for text in text_list]

    def fit(self, X: pd.DataFrame, y=None):
        missing_cols = [col for col in [self.ingredients_col, self.steps_col] if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in the DataFrame")
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[List[List[str]], List[List[str]]]:
        X = X.dropna(subset=[self.ingredients_col, self.steps_col])
        ingredients_lists = []
        steps_lists = []
        ingredients_counter = defaultdict(int)

        for _, row in X.iterrows():
            ingredients = self._extract_ingredients(row)
            steps = self._extract_steps(row)
            
            for ing in ingredients:
                ingredients_counter[ing] += 1
                
            ingredients_lists.append(ingredients)
            steps_lists.append(steps)

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

    def fit_transform(self, X: pd.DataFrame, y=None) -> Tuple[List[List[str]], List[List[str]]]:
        return self.fit(X).transform(X)