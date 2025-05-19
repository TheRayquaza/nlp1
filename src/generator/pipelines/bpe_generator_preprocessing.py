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

class BPEGeneratorPreprocessing:
    def __init__(self,
                 ingredients_col='ingredients_text',
                 steps_col='steps_string_standardize',
                 drop_uncommon=None,
                 vocab_size=256):
        self.ingredients_col = ingredients_col
        self.steps_col = steps_col
        self.stemmer = LancasterStemmer()
        self.drop_uncommon = drop_uncommon
        self.vocab_size = vocab_size
        
        self.vocab = []
        self.merges = {}
        self.word_freqs = defaultdict(int)
        self.splits = {}

    def get_savepath(self, prefix: str):
        return f"{prefix}bpe_preprocessing_{self.vocab_size}_vocab_size";

    def _extract_ingredients(self, row) -> List[str]:
        if pd.isna(row[self.ingredients_col]):
            print("  - No ingredients found (NaN value)")
            return []

        ingredients = ast.literal_eval(row[self.ingredients_col])
        stemmed = [self.stemmer.stem(ing) for ing in ingredients if ing]
        return stemmed

    def _extract_steps(self, row) -> List[str]:
        if pd.isna(row[self.steps_col]):
            print("  - No steps found (NaN value)")
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
        print(f"\nCollecting corpus from {len(X)} rows...")
        corpus = []
        for idx, row in X.iterrows():
            if idx % 100 == 0 and idx > 0:
                print(f"  - Processed {idx} rows, corpus size: {len(corpus)}")
            
            ingredients = self._extract_ingredients(row)
            steps = self._extract_steps(row)
            
            if ingredients:
                corpus.extend(ingredients)
            if steps:
                corpus.extend(steps)
                
        print(f"Final corpus size: {len(corpus)} tokens")
        print(f"Sample corpus entries: {corpus[:5]}")
        return corpus

    def _train_bpe(self, corpus: List[str]):
        print("\nTraining BPE tokenizer...")
        print("Building initial word frequencies...")
        for text in corpus:
            self.word_freqs[text] += 1
        
        print(f"Unique words in corpus: {len(self.word_freqs)}")
        print(f"Top 5 most common words: {Counter(self.word_freqs).most_common(5)}")
        
        print("\nBuilding initial alphabet...")
        alphabet = set()
        for word in self.word_freqs.keys():
            for letter in word:
                alphabet.add(letter)
        
        print(f"Alphabet size: {len(alphabet)}")
        
        self.vocab = ["<s>", "</s>", "<RECIPE>"] + sorted(list(alphabet))
        print(f"Initial vocabulary size: {len(self.vocab)}")
        
        print("\nInitializing splits (words into characters)...")
        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}
        
        # Apply BPE algorithm
        print("\nStarting BPE merge operations...")
        iteration = 0
        while len(self.vocab) < self.vocab_size:
            iteration += 1
            print(f"\nIteration {iteration}: Current vocab size = {len(self.vocab)}")
            
            print("  Computing pair frequencies...")
            pair_freqs = self._compute_pair_freqs()
            if not pair_freqs:
                print("  No more pairs to merge, stopping early")
                break
                
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            best_pair_freq = pair_freqs[best_pair]
            print(f"  Best pair: ('{best_pair[0]}', '{best_pair[1]}') with frequency {best_pair_freq}")
            
            print(f"  Merging pair '{best_pair[0]}' + '{best_pair[1]}'...")
            self.splits = self._merge_pair(best_pair[0], best_pair[1])
            
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])
            
            if iteration % 100 == 0:
                print(f"  Current vocab size: {len(self.vocab)}")
                print(f"  Latest 5 tokens: {self.vocab[-5:]}")
        
        print(f"\nBPE training complete. Final vocabulary size: {len(self.vocab)}")
        print(f"Sample vocabulary entries: {self.vocab[:10]}...{self.vocab[-10:]}")
    
    def _compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        
        return pair_freqs
    
    def _merge_pair(self, a: str, b: str):
        new_splits = {}
        for word, split in self.splits.items():
            if len(split) == 1:
                new_splits[word] = split
                continue

            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                    new_split.append(a + b)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        return new_splits

    def flatmap_tokens(self, nested_lists: List[List[List[str]]]) -> List[List[str]]:
        print(f"\nFlattening nested token structure...")
        result = []

        for recipe_idx, recipe_tokens in enumerate(nested_lists):
            if recipe_idx % 100 == 0 and recipe_idx > 0:
                print(f"  - Flattened {recipe_idx} recipes")
                
            flattened = []
            for item_tokens in recipe_tokens:
                flattened.extend(item_tokens)
            
            result.append(flattened)
            
        print(f"Flattening complete. Converted {len(nested_lists)} nested lists to flat token lists.")
        return result

    def tokenize(self, text: str) -> List[str]:
        chars = [c for c in text]
        
        i = 0
        merges_applied = 0
        while i < len(chars) - 1:
            pair = (chars[i], chars[i + 1])
            if pair in self.merges:
                chars = chars[:i] + [self.merges[pair]] + chars[i+2:]
                merges_applied += 1
            else:
                i += 1
        
        return chars

    def tokenize_list(self, text_list: List[str]) -> List[List[str]]:
        return [self.tokenize(text) for text in text_list]

    def fit(self, X: pd.DataFrame, y=None):
        print(f"\n{'='*50}")
        print(f"FITTING BPE TOKENIZER ON {len(X)} SAMPLES")
        print(f"{'='*50}")
        
        missing_cols = [col for col in [self.ingredients_col, self.steps_col] if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in the DataFrame")

        corpus = self._collect_corpus(X)
        
        self._train_bpe(corpus)

        print(f"\nFit complete!")
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[List[List[str]], List[List[str]]]:
        print(f"\n{'='*50}")
        print(f"TRANSFORMING {len(X)} SAMPLES")
        print(f"{'='*50}")
        
        X = X.dropna(subset=[self.ingredients_col, self.steps_col])
        print(f"After dropping NaN values: {len(X)} samples remain")
        
        ingredients_lists = []
        steps_lists = []
        ingredients_counter = defaultdict(int)

        print("\nExtracting ingredients and steps...")
        for idx, row in X.iterrows():
            if idx % 1000 == 0 and idx > 0:
                print(f"  - Processed {idx} rows")
                
            ingredients = self._extract_ingredients(row)
            steps = self._extract_steps(row)
            
            for ing in ingredients:
                ingredients_counter[ing] += 1
                
            ingredients_lists.append(ingredients)
            steps_lists.append(steps)

        print(f"\nExtracted {len(ingredients_lists)} ingredient lists and {len(steps_lists)} step lists")
        
        if self.drop_uncommon is not None:
            print(f"\nFiltering uncommon ingredients (threshold: {self.drop_uncommon})...")
            total_ingredients = sum(len(ing_list) for ing_list in ingredients_lists)
            print(f"Total ingredients before filtering: {total_ingredients}")
            
            uncommon_ingredients = {
                ing for ing, count in ingredients_counter.items()
                if count < self.drop_uncommon
            }
            
            print(f"Found {len(uncommon_ingredients)} uncommon ingredients to filter")
            print(f"Examples of uncommon ingredients: {list(uncommon_ingredients)[:5]}")
    
            filtered_ingredients_lists = []
            filtered_steps_lists = []
            for ingredients, steps in zip(ingredients_lists, steps_lists):
                if not any(ing in uncommon_ingredients for ing in ingredients):
                    filtered_ingredients_lists.append(ingredients)
                    filtered_steps_lists.append(steps)
    
            print(f"After filtering: {len(filtered_ingredients_lists)} samples remain")
            ingredients_lists = filtered_ingredients_lists
            steps_lists = filtered_steps_lists

        print("\nTokenizing ingredients and steps...")
        tokenized_ingredients_lists = [self.tokenize_list(ingredients) for ingredients in ingredients_lists]
        tokenized_steps_lists = [self.tokenize_list(steps) for steps in steps_lists]
        
        print("\nFlattening token structures...")
        flat_ingredients_lists = self.flatmap_tokens(tokenized_ingredients_lists)
        flat_steps_lists = self.flatmap_tokens(tokenized_steps_lists)
        
        print(f"\nFinal result: {len(flat_ingredients_lists)} ingredient token lists, {len(flat_steps_lists)} step token lists")
        if flat_ingredients_lists and flat_steps_lists:
            print(f"Example ingredients tokens (first recipe): {flat_ingredients_lists[0][:10]}...")
            print(f"Example steps tokens (first recipe): {flat_steps_lists[0][:10]}...")
        
        print("\nTransform complete!")
        return flat_ingredients_lists, flat_steps_lists

    def fit_transform(self, X: pd.DataFrame, y=None) -> Tuple[List[List[str]], List[List[str]]]:
        print("\nPerforming fit_transform...")
        return self.fit(X).transform(X)

    def save_in_dir(self, directory: str):
        self.save(self.get_savepath(directory))

    def save(self, path: str):
        print(f"\nSaving tokenizer to {path}...")
        tokenizer_data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'word_freqs': dict(self.word_freqs),
            'splits': self.splits
        }
        with open(path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        print("Tokenizer saved successfully!")
    
    def load(self, path: str):
        print(f"\nLoading tokenizer from {path}...")
        with open(path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        self.vocab = tokenizer_data['vocab']
        self.merges = tokenizer_data['merges']
        self.word_freqs = defaultdict(int)
        self.word_freqs.update(tokenizer_data['word_freqs'])
        self.splits = tokenizer_data['splits']
        print(f"Tokenizer loaded successfully!")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")
