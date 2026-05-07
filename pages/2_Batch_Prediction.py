import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import sys
import os

# Asegurar que utils sea importable
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from utils.model_loader import load_models_and_scaler
from utils.styles import apply_custom_design

st.set_page_config(page_title="Predicción por Lotes", page_icon="📁", layout="wide")
apply_custom_design()

st.title("📁 Predicción por Lotes y Evaluación")

scaler, lr_model, nn_model = load_models_and_scaler()

if not scaler or not lr_model or not nn_model:
    st.warning("Los modelos no están disponibles. Asegúrate de que el script de entrenamiento haya terminado con éxito.")
    st.stop()

st.markdown("""
Sube un archivo de datos (CSV o delimitado por espacios) que contenga **22 columnas** (21 características + 1 etiqueta real).
Por ejemplo, puedes subir el archivo `ann-test.data` de tu directorio original.
""")

model_choice = st.radio("Seleccione el modelo predictivo a evaluar:", ("Red Neuronal (MLP)", "Regresión Logística"))
model_to_use = nn_model if model_choice == "Red Neuronal (MLP)" else lr_model

uploaded_file = st.file_uploader("Elige un archivo de datos", type=['csv', 'data', 'txt'])

if uploaded_file is not None:
    try:
        # Intentar leer primero como CSV
        try:
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[1] < 22:
                # Si falló porque no es CSV con comas, volver al inicio e intentar con espacios
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
            
        st.write(f"Datos cargados: {df.shape[0]} filas y {df.shape[1]} columnas.")
        
        if df.shape[1] < 22:
            st.error("El archivo debe contener al menos 22 columnas (21 características + 1 etiqueta).")
        else:
            X = df.iloc[:, :21]
            y_true = df.iloc[:, 21]
            
            # Nombrar columnas para el scaler
            feature_names = [f'feature_{i}' for i in range(1, 22)]
            X.columns = feature_names
            
            # Escalar
            X_scaled = scaler.transform(X)
            
            # Predecir
            y_pred = model_to_use.predict(X_scaled)
            
            st.subheader("Resultados de Predicción")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Matriz de Confusión**")
                cm = confusion_matrix(y_true, y_pred)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='mako', ax=ax, cbar=False)
                ax.set_xlabel('Predicción', color='#64FFDA')
                ax.set_ylabel('Realidad', color='#64FFDA')
                ax.tick_params(colors='#64FFDA')
                ax.set_title(f'Matriz de Confusión - {model_choice}', color='#64FFDA', fontweight='bold')
                fig.patch.set_facecolor('#112240')
                ax.set_facecolor('#112240')
                st.pyplot(fig)
                
            with col2:
                st.write("**Reporte de Clasificación**")
                report = classification_report(y_true, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.style.format("{:.3f}"))
                
    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
