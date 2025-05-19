import pandas as pd
import numpy as np
import nltk
import ast
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List
import logging
import os

class TFIDFPreprocessing:
    def __init__(self, step_max_features=200, ingredients_max_features=200):
        self.logger = logging.getLogger(__name__)
        self.steps_vectorizer = TfidfVectorizer(max_features=step_max_features, stop_words='english')
        self.ingredients_vectorizer = TfidfVectorizer(max_features=ingredients_max_features, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.valid_cuisines = {}

    def fit(self, data: pd.DataFrame):
        data.dropna(subset=['cuisine', 'steps_string_standardize', 'ingredients_text'], inplace=True)

        self.logger.info("Extracting primary cuisine...")
        self.valid_cuisines = set(data['cuisine'].unique())

        self.logger.info("Fitting TF-IDF vectorizers...")
        self.steps_vectorizer.fit(data["steps_string_standardize"])
        self.ingredients_vectorizer.fit(data["ingredients_text"])

        self.logger.info("Fitting label encoder...")
        self.label_encoder.fit(data['cuisine'])

        return self

    def transform(self, data) -> Tuple[np.ndarray, np.ndarray]:
        data.dropna(subset=['cuisine', 'steps_string_standardize', 'ingredients_text'], inplace=True)

        self.logger.info("Starting transform process...")        
        if self.valid_cuisines is None:
            raise ValueError("The preprocessor has not been fitted. Call fit() first.")

        self.logger.info(f"Removing invalid cuisines: {data['cuisine'].isin(self.valid_cuisines)}")
        data = data[data['cuisine'].isin(self.valid_cuisines)].copy()

        # Transform text data using fitted vectorizers
        self.logger.info("Applying TF-IDF transformations...")
        steps_features = self.steps_vectorizer.transform(data["steps_string_standardize"]).toarray()
        ingredients_features = self.ingredients_vectorizer.transform(data["ingredients_text"]).toarray()

        # Extract numerical features
        numerical_features = data[['n_steps', 'n_ingredients']].values
        
        # Combine all features
        self.logger.info("Building feature matrix...")
        feature_matrices = [
            steps_features,
            ingredients_features,
            numerical_features
        ]

        X = np.hstack(feature_matrices)
        y = self.label_encoder.transform(data['cuisine'])
        return X, y

    def fit_transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(data)
        return self.transform(data)
    
    def save_pipeline(self, directory="./preprocessor"):
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, "steps_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.steps_vectorizer, f)

        with open(os.path.join(directory, "ingredients_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.ingredients_vectorizer, f)

        with open(os.path.join(directory, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(os.path.join(directory, "valid_cuisines.pkl"), "wb") as f:
            pickle.dump(self.valid_cuisines, f)

        self.logger.info(f"Preprocessor saved to {directory}")
        
    def load_pipeline(self, directory="./preprocessor"):
        with open(os.path.join(directory, "steps_vectorizer.pkl"), "rb") as f:
            self.steps_vectorizer = pickle.load(f)

        with open(os.path.join(directory, "ingredients_vectorizer.pkl"), "rb") as f:
            self.ingredients_vectorizer = pickle.load(f)

        with open(os.path.join(directory, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(os.path.join(directory, "valid_cuisines.pkl"), "rb") as f:
            self.valid_cuisines = pickle.load(f)

        self.logger.info(f"Preprocessor loaded from {directory}")
        return self
