import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import joblib

nltk.download('punkt_tab')
nltk.download('stopwords')
def get_mae_by_recipe_length(y_true, y_pred, threshold_long=60, threshold_fast=30):

    # Create masks for fast and long recipes
    fast_recipes_mask = y_true <= threshold_fast
    long_recipes_mask = y_true > threshold_long
    
    # Calculate MAE for each category
    mae_fast = mean_absolute_error(y_true[fast_recipes_mask], y_pred[fast_recipes_mask]) if any(fast_recipes_mask) else None
    mae_long = mean_absolute_error(y_true[long_recipes_mask], y_pred[long_recipes_mask]) if any(long_recipes_mask) else None
    
    return {
        "MAE_fast_recipes": mae_fast,
        "MAE_long_recipes": mae_long
    }

def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae_by_recipe_length = get_mae_by_recipe_length(y_true, y_pred)
    return {
        "MAE global": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "MAE_fast_recipes": mae_by_recipe_length["MAE_fast_recipes"],
        "MAE_long_recipes": mae_by_recipe_length["MAE_long_recipes"]
    }

def save_feature_matrices(path, matrices):
    joblib.dump(matrices, path)

def load_feature_matrices(path):
    return joblib.load(path)

# Functions for tfidf
def build_features_tfidf(data):
    dir_path = "models/features"
    save_path = f"{dir_path}/tfidf_matrice_features.pkl"
    os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(save_path):
        print(f"Loading cached TF_IDF features from '{save_path}'...")
        return load_feature_matrices(save_path)

    print("\nBuilding TF_IDF features matrices...")
    steps_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    steps_features = steps_vectorizer.fit_transform(data["steps_string_standardize"])

    ingredients_vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
    ingredients_features = ingredients_vectorizer.fit_transform(data["ingredients_text"])

    tags_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tags_features = tags_vectorizer.fit_transform(data["tags_text"])
    
    numerical_features = data[['n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar', 'sodium', 'saturated_fat', 'carbohydrates']].values

    feature_matrices = [steps_features.toarray(), ingredients_features.toarray(),
                    tags_features.toarray(),
                   numerical_features]
    save_feature_matrices(save_path, feature_matrices)
    print("Finish building TF_IDF features matrices...")
    return feature_matrices


# Functions for word2Vec

def preprocess_text(text):
    """Tokenize text and remove stopwords"""
    if not isinstance(text, str):
        return []
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    return tokens

def build_word2vec_model(tokenized_texts, vector_size=100, window=5, min_count=1, workers=max(1, int(os.cpu_count() * 0.70))):
    """Train a Word2Vec model on the tokenized texts"""
    model = Word2Vec(sentences=tokenized_texts, 
                    vector_size=vector_size, 
                    window=window, 
                    min_count=min_count, 
                    workers=workers)
    model.train(tokenized_texts, total_examples=len(tokenized_texts), epochs=10)
    return model

def get_document_vector(tokenized_doc, word2vec_model, vector_size):
    """Create a document vector by averaging word vectors"""
    doc_vector = np.zeros(vector_size)
    count = 0
    
    for word in tokenized_doc:
        if word in word2vec_model.wv:
            doc_vector += word2vec_model.wv[word]
            count += 1
    
    if count > 0:
        doc_vector /= count
    
    return doc_vector

def build_features_word2vec(data):
    dir_path = "models/features"
    save_path = f"{dir_path}/word2vec_matrice_features.pkl"
    os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(save_path):
        print(f"Loading cached Word2Vec features from '{save_path}'...")
        return load_feature_matrices(save_path)
    print("\nBuilding Word2Vec features matrices...")

    # Tokenize texts
    steps_tokenized = data["steps_string_standardize"].apply(preprocess_text).tolist()
    ingredients_tokenized = data["ingredients_text"].apply(preprocess_text).tolist()
    tags_tokenized = data["tags_text"].apply(preprocess_text).tolist()
    
    # Build Word2Vec models
    steps_w2v = build_word2vec_model(steps_tokenized, vector_size=400)
    ingredients_w2v = build_word2vec_model(ingredients_tokenized, vector_size=150)
    tags_w2v = build_word2vec_model(tags_tokenized, vector_size=100)
    
    # Create document vectors
    steps_features = np.array([get_document_vector(doc, steps_w2v, 400) for doc in steps_tokenized])
    ingredients_features = np.array([get_document_vector(doc, ingredients_w2v, 150) for doc in ingredients_tokenized])
    tags_features = np.array([get_document_vector(doc, tags_w2v, 100) for doc in tags_tokenized])
    
    # Numerical features
    numerical_features = data[['n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar', 'sodium', 'saturated_fat', 'carbohydrates']].values
    
    feature_matrices = [steps_features, ingredients_features, tags_features, numerical_features]
    save_feature_matrices(save_path, feature_matrices)
    print("Finish building Word2Vec features matrices...")
    return feature_matrices
