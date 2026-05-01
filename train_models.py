import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_data(file_path):
    # El archivo está separado por espacios y puede tener múltiples espacios
    # La última columna es la clase, las 21 anteriores son las features.
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

def main():
    print("Cargando datos de entrenamiento y prueba...")
    X_train, y_train = load_data('thyroid+disease/ann-train.data')
    X_test, y_test = load_data('thyroid+disease/ann-test.data')
    
    # Nombres de las features genéricos ya que no hay cabecera en el archivo
    feature_names = [f'feature_{i}' for i in range(1, 22)]
    X_train.columns = feature_names
    X_test.columns = feature_names

    print("Escalando datos...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Guardar scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    print("Entrenando Regresión Logística...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    print("Evaluando Regresión Logística...")
    y_pred_lr = lr_model.predict(X_test_scaled)
    print("Accuracy LR:", accuracy_score(y_test, y_pred_lr))
    
    joblib.dump(lr_model, 'models/logistic_regression.pkl')
    
    print("Entrenando Red Neuronal (MLP)...")
    nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    nn_model.fit(X_train_scaled, y_train)
    
    print("Evaluando Red Neuronal...")
    y_pred_nn = nn_model.predict(X_test_scaled)
    print("Accuracy NN:", accuracy_score(y_test, y_pred_nn))
    
    joblib.dump(nn_model, 'models/neural_network.pkl')
    
    print("Modelos entrenados y guardados en la carpeta 'models/'.")

if __name__ == '__main__':
    main()
