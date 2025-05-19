import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from train_models import train_models
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.model_selection import train_test_split
import numpy as np
from preprocessing import process_data

ps = PorterStemmer()


def train_classification_model():
    data_path = '../data/processed/preprocessed_recipe.csv'
    if os.path.isfile(data_path):
        print("Loading preprocessed data...")
        data = pd.read_csv(data_path)
    else:
        print("File not found, preprocessing the dataset...")
        data = process_data(save=True)
        print("Loading preprocessed data...")

    df_results = train_models(data)
    print("\nBenchmark Results:")
    print(df_results)

train_classification_model()
