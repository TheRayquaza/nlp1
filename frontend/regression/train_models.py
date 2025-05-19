import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

from frontend.regression.utils.utils_train import build_features_tfidf, build_features_word2vec, get_metrics

def predict_cooking_time(model_key, recipe_data, feature_matrices, train_test_data):
    X_train, X_test, _, _, X_train_indices, X_test_indices = train_test_data
    
    # Find the index of the recipe in the dataset
    recipe_idx = recipe_data.name
    
    # Extract embedding type from model_key
    embedding = "tf" if model_key.endswith("tf") else "w2v"
    
    # Check if the recipe is in the training or test set
    if recipe_idx in X_train_indices:
        # Find the position of the recipe index in X_train_indices
        train_pos = list(X_train_indices).index(recipe_idx)
        feature_vector = X_train[train_pos:train_pos+1]
    elif recipe_idx in X_test_indices:
        # Find the position of the recipe index in X_test_indices
        test_pos = list(X_test_indices).index(recipe_idx)
        feature_vector = X_test[test_pos:test_pos+1]
    else:
        # For recipes not in either set, we need a different approach
        return recipe_data['minutes']  # Fallback to actual time as we can't predict
    
    # Load the appropriate saved model based on model_key
    model_type = model_key.split("_")[0]  # Extract model type (lr, bayesian, forward_nn, gru)
    
    # Define model paths based on your saving convention
    model_dir = "models/saved"
    
    if model_type == "lr":
        model_path = f"{model_dir}/lr_model_{embedding}.pkl"
        model = joblib.load(model_path)
        prediction = model.predict(feature_vector)[0]

    elif model_type == "bayesian":
        model_path = f"{model_dir}/bayes_model_{embedding}.pkl"
        model = joblib.load(model_path)
        prediction = model.predict(feature_vector)[0]
            
    elif model_type == "forward":
        model_path = f"{model_dir}/bayes_model_{embedding}.keras"

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        feature_vector_scaled = scaler.transform(feature_vector)
        
        model = load_model(model_path)
        prediction = model.predict(feature_vector_scaled).flatten()[0]
    
    # elif model_type == "gru":
        
    else:
        # Default fallback
        prediction = recipe_data['minutes']
    
    # Ensure prediction is positive
    prediction = max(0, prediction)
    
    return prediction


def train_test_split_custom(data, feature_matrice):
    X = np.hstack(feature_matrice)
    y = data["minutes"].values

    # Make sure to return the indices as well
    X_train, X_test, y_train, y_test, X_train_indices, X_test_indices = (
        train_test_split(X, y, data.index, test_size=0.2, random_state=42)
    )
    
    return X_train, X_test, y_train, y_test, X_train_indices, X_test_indices


def create_feature_matrices(data, embedding):
    if embedding == "w2v":
        return build_features_word2vec(data)
    elif embedding == "tf":
        return build_features_tfidf(data)
    else:
        return []

## FEEDFORWARD NEURAL NETWORKS FUNCTIONS
def train_feedforward(embedding, X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    dir_path = "models/saved"
    os.makedirs(dir_path, exist_ok=True)
    model_path = f"{dir_path}/bayes_model_{embedding}.keras"

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

        model = Sequential(
            [
                # Input layer
                # l2 => to prevent overfitting
                Dense(
                    128,
                    activation="relu",
                    input_shape=(X_train.shape[1],),
                    kernel_regularizer=l2(0.001),
                ),
                BatchNormalization(),
                Dropout(0.3),
                # Hidden layers
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                # Output layer (single neuron for regression)
                Dense(1),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mean_squared_error",  # MSE for regression
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        model.fit(
            X_train_scaled,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )
        model.save(model_path)
    
    y_pred = model.predict(X_test_scaled).flatten()
    return get_metrics(y_test, y_pred)

def forward_nn(embedding, X_train, X_test, y_train, y_test):

    print(f"\n{embedding} - FeedForward NN Start")
    feedforward_metrics = train_feedforward(embedding, X_train, X_test, y_train, y_test)
    print(f"{embedding} - FeedForward NN Done")

    return feedforward_metrics

## NAÏVE BAYES FUNCTIONS
def train_bayes(embedding, X_train, X_test, y_train, y_test):
    dir_path = "models/saved"
    os.makedirs(dir_path, exist_ok=True)
    model_path = f"{dir_path}/bayes_model_{embedding}.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = BayesianRidge()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    return get_metrics(y_test, y_pred)

def bayesian(embedding, X_train, X_test, y_train, y_test):

    print(f"\n{embedding} - Bayesian Ridge Start")
    # Bayesian Ridge
    bayesian_metrics = train_bayes(
        embedding, X_train, X_test, y_train, y_test
    )
    print(f"{embedding} - Bayesian Ridge Done")

    return bayesian_metrics


## LINEAR REGRESSION FUNCTIONS
def train_linear_regression(embedding, X_train, X_test, y_train, y_test):
    dir_path = "models/saved"
    os.makedirs(dir_path, exist_ok=True)
    model_path = f"{dir_path}/lr_model_{embedding}.pkl"
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    return get_metrics(y_test, y_pred)

def linear(embedding, X_train, X_test, y_train, y_test):

    print(f"\n{embedding} - Linear Regression Start")
    
    linear_metrics = train_linear_regression(
        embedding, X_train, X_test, y_train, y_test
    )

    print(f"{embedding} - Linear Regression Done")

    return linear_metrics


def select_model(model: str, X_train, X_test, y_train, y_test):
    match model:
        # TF-IDF Models
        case "lr_tf":
            return linear("tf", X_train, X_test, y_train, y_test)
        case "bayesian_tf":
            return bayesian("tf", X_train, X_test, y_train, y_test)
        case "forward_nn_tf":
            return forward_nn("tf", X_train, X_test, y_train, y_test)

        # Word2Vec Models
        case "lr_w2v":
            return linear("w2v", X_train, X_test, y_train, y_test)
        case "bayesian_w2v":
            return bayesian("w2v", X_train, X_test, y_train, y_test)
        case "forward_nn_w2v":
            return forward_nn("w2v", X_train, X_test, y_train, y_test)

        # Default
        case _:
            print(f"ERROR - model {model} should not exists")
            return {"weights": []}
    return []
