import time
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from utils.utils_train import build_features_tfidf, build_features_word2vec, get_metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

def train_simple_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return get_metrics(y_test, y_pred)

def train_feedforward(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    model = Sequential([
            # Input layer
            # l2 => to prevent overfitting
            Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer (single neuron for regression)
            Dense(1)
        ])

    model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'  # MSE for regression
    )

    early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
    )

    history = model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
    )
    y_pred = model.predict(X_test_scaled).flatten()
    return get_metrics(y_test, y_pred)


def train_models(data):
    print("Building Feature Matrices")

    feature_matrices = [(build_features_tfidf(data), 'TF-IDF'), (build_features_word2vec(data), 'Word2Vec')]
    results = []
    
    print("Feature Matrices are ready")
    for feature_matrice, embedding in feature_matrices:
        X = np.hstack(feature_matrice)
        y = data["minutes"].values

        X_train, X_test, y_train, y_test, X_train_indices, X_test_indices = train_test_split(
            X, y, data.index, test_size=0.2, random_state=42
        )
        print(f"\nRunning tests for {embedding}")
        # Linear Regression
        print(f"\n{embedding} - Linear Regression Start")
        start = time.time()
        linear_metrics = train_simple_model(LinearRegression(), X_train, y_train, X_test, y_test)
        linear_metrics.update({
            "model": "Linear Regression",
            "train_time_in_seconds": round(time.time() - start, 2),
            "embedding_type": embedding
        })
        results.append(linear_metrics)
        print(f"{embedding} - Linear Regression Done")
        print(f"\n{embedding} - Bayesian Ridge Start")
        # Bayesian Ridge
        start = time.time()
        bayesian_metrics = train_simple_model(BayesianRidge(), X_train, y_train, X_test, y_test)
        bayesian_metrics.update({
            "model": "Bayesian Ridge",
            "train_time_in_seconds": round(time.time() - start, 2),
            "embedding_type": embedding
        })
        results.append(bayesian_metrics)
        print(f"{embedding} - Bayesian Ridge Done")
        # Feedforward NN
        print(f"\n{embedding} - FeedForward NN Start")
        start = time.time()
        feedforward_metrics = train_feedforward(X_train, y_train, X_test, y_test)
        feedforward_metrics.update({
            "model": "Feedforward NN",
            "train_time_in_seconds": round(time.time() - start, 2),
            "embedding_type": embedding
        })
        results.append(feedforward_metrics)
        print(f"{embedding} - FeedForward NN Done")
    return pd.DataFrame(results).set_index("model")
