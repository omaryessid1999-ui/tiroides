import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 22:
            row = [float(x) for x in parts[:22]]
            data.append(row)
            
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def train_and_get_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'thyroid+disease', 'ann-train.data')
    test_path = os.path.join(base_dir, 'thyroid+disease', 'ann-test.data')
    
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)
    
    # Combinar los datasets
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = pd.concat([y_train, y_test], ignore_index=True)
    
    feature_names = [f'feature_{i}' for i in range(1, 22)]
    X_combined.columns = feature_names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

    # --- 1. RED NEURONAL (MLP) ---
    mlp = MLPClassifier(max_iter=500) # Reducido para no bloquear la app mucho tiempo
    # Usando parámetros más ligeros para que el entrenamiento no sea eterno al cargar la página
    param_grid = {
        "hidden_layer_sizes": [(8,), (16,)],
        "activation": ["relu"],
        "alpha": [0.01],
    }

    grid_search = RandomizedSearchCV(
        mlp,
        param_grid,
        cv=2,
        n_iter=2,
        n_jobs=-1,
        scoring="accuracy",
        random_state=42
    )
    
    grid_search.fit(X_scaled, y_combined)
    best_model = grid_search.best_estimator_

    # Hacer predicciones sobre el mismo set para mostrar algo si es necesario (accuracy de entrenamiento)
    y_pred_nn = best_model.predict(X_scaled)
    joblib.dump(best_model, os.path.join(models_dir, 'neural_network.pkl'))

    # --- 2. REGRESIÓN LOGÍSTICA ---
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_scaled, y_combined)
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression.pkl'))
    
    return scaler, lr_model, best_model

def main():
    print("Iniciando entrenamiento y combinando datasets...")
    train_and_get_models()
    print("¡Modelos entrenados y guardados en la carpeta 'models/' con éxito!")

if __name__ == '__main__':
    main()
