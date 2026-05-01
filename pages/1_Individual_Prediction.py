import streamlit as st
import pandas as pd
import sys
import os

# Asegurar que utils sea importable
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from utils.model_loader import load_models_and_scaler

st.set_page_config(page_title="Predicción Individual", page_icon="👤", layout="wide")

st.title("👤 Predicción Individual de Tiroides")

scaler, lr_model, nn_model = load_models_and_scaler()

if not scaler or not lr_model or not nn_model:
    st.warning("Los modelos no están disponibles. Asegúrate de que el script de entrenamiento haya terminado con éxito.")
    st.stop()

st.markdown("""
Ingrese los 21 parámetros clínicos del paciente. (Las descripciones originales del dataset no especifican los nombres de las columnas, por lo que usaremos nombres genéricos).
""")

# Seleccionar modelo
model_choice = st.radio("Seleccione el modelo predictivo:", ("Red Neuronal (MLP)", "Regresión Logística"))
model_to_use = nn_model if model_choice == "Red Neuronal (MLP)" else lr_model

with st.form("prediction_form"):
    st.subheader("Datos del Paciente")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        f1 = st.number_input("Feature 1 (Continuo - ej. Edad normalizada)", value=0.5, format="%.3f")
        f2 = st.selectbox("Feature 2 (Binario)", options=[0, 1])
        f3 = st.selectbox("Feature 3 (Binario)", options=[0, 1])
        f4 = st.selectbox("Feature 4 (Binario)", options=[0, 1])
        f5 = st.selectbox("Feature 5 (Binario)", options=[0, 1])
        f6 = st.selectbox("Feature 6 (Binario)", options=[0, 1])
        f7 = st.selectbox("Feature 7 (Binario)", options=[0, 1])
        
    with col2:
        f8 = st.selectbox("Feature 8 (Binario)", options=[0, 1])
        f9 = st.selectbox("Feature 9 (Binario)", options=[0, 1])
        f10 = st.selectbox("Feature 10 (Binario)", options=[0, 1])
        f11 = st.selectbox("Feature 11 (Binario)", options=[0, 1])
        f12 = st.selectbox("Feature 12 (Binario)", options=[0, 1])
        f13 = st.selectbox("Feature 13 (Binario)", options=[0, 1])
        f14 = st.selectbox("Feature 14 (Binario)", options=[0, 1])
        
    with col3:
        f15 = st.selectbox("Feature 15 (Binario)", options=[0, 1])
        f16 = st.selectbox("Feature 16 (Binario)", options=[0, 1])
        f17 = st.number_input("Feature 17 (Continuo - TSH, etc.)", value=0.001, format="%.5f")
        f18 = st.number_input("Feature 18 (Continuo)", value=0.020, format="%.5f")
        f19 = st.number_input("Feature 19 (Continuo)", value=0.100, format="%.5f")
        f20 = st.number_input("Feature 20 (Continuo)", value=0.090, format="%.5f")
        f21 = st.number_input("Feature 21 (Continuo)", value=0.110, format="%.5f")

    submitted = st.form_submit_button("Predecir")

if submitted:
    # Crear DataFrame con el registro
    input_data = [[
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16,
        f17, f18, f19, f20, f21
    ]]
    feature_names = [f'feature_{i}' for i in range(1, 22)]
    df_input = pd.DataFrame(input_data, columns=feature_names)
    
    # Escalar
    scaled_input = scaler.transform(df_input)
    
    # Predecir
    prediction = model_to_use.predict(scaled_input)[0]
    
    # Map classes
    class_map = {
        1: "Clase 1 (Hipertiroidismo)", 
        2: "Clase 2 (Hipotiroidismo)", 
        3: "Clase 3 (Normal)"
    }
    pred_str = class_map.get(prediction, f"Clase {prediction}")
    
    st.success(f"### Resultado Predictivo: {pred_str}")
    st.info("Nota: Las clases 1 y 2 indican anomalía, la clase 3 suele indicar estado normal según el dataset original.")
