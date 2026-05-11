import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import random

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from utils.model_loader import load_models_and_scaler
from utils.styles import apply_custom_design
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ── Funciones de Referencia General ──
def load_official_test_data():
    test_path = os.path.join(base_dir, 'thyroid+disease', 'ann-test.data')
    if not os.path.exists(test_path):
        return None, None
    with open(test_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 22:
            data.append([float(x) for x in parts[:22]])
    df = pd.DataFrame(data)
    return df.iloc[:, :21], df.iloc[:, 21]

def plot_cm(cm, title, labels=["Hiper", "Hipo", "Norm"]):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako', ax=ax, cbar=False,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicción', color='#64FFDA', fontsize=8)
    ax.set_ylabel('Realidad', color='#64FFDA', fontsize=8)
    ax.set_title(title, color='#64FFDA', fontweight='bold', fontsize=10)
    fig.patch.set_facecolor('#112240')
    ax.set_facecolor('#112240')
    ax.tick_params(colors='#64FFDA', labelsize=7)
    plt.tight_layout()
    return fig

st.set_page_config(page_title="Diagnóstico Individual", page_icon="👤", layout="wide")
apply_custom_design()

# ── Definición de Features ──
FEATURES = [
    ("Edad (normalizada)", "continuous"), ("Sexo (1=M, 0=F)", "binary"),
    ("On Thyroxine", "binary"), ("Query on Thyroxine", "binary"),
    ("On Antithyroid", "binary"), ("Sick", "binary"), ("Pregnant", "binary"),
    ("Thyroid Surgery", "binary"), ("I131 Treatment", "binary"),
    ("Query Hypothyroid", "binary"), ("Query Hyperthyroid", "binary"),
    ("Lithium", "binary"), ("Goitre", "binary"), ("Tumor", "binary"),
    ("Hypopituitary", "binary"), ("Psych", "binary"),
    ("TSH", "continuous"), ("T3", "continuous"), ("TT4", "continuous"),
    ("T4U", "continuous"), ("FTI", "continuous")
]

# ── Presentación ──
st.title("👤 Panel de Diagnóstico Clínico")
st.markdown("""
### Bienvenido al Sistema de Inteligencia Artificial Tiroidea
Esta sección permite realizar un análisis individualizado. Ingrese los biomarcadores del paciente para obtener un diagnóstico basado en modelos de aprendizaje profundo y regresión logística.
""")

scaler, lr_model, nn_model = load_models_and_scaler()
if not scaler or not lr_model or not nn_model:
    st.stop()

# ── Selección de Modelo ──
st.sidebar.header("Configuración")
model_choice = st.sidebar.radio("Modelo:", ("Red Neuronal", "Regresión Logística"))
model_to_use = nn_model if model_choice == "Red Neuronal" else lr_model

# ── Botón Aleatorio ──
if st.button("🎲 Generar Datos Aleatorios (Paciente Nuevo)"):
    for i, (_, ftype) in enumerate(FEATURES):
        key = f"f_{i}"
        if ftype == "continuous":
            new_val = round(random.uniform(0.001, 0.5), 5)
        else:
            new_val = random.choice([0, 1])
        st.session_state[key] = new_val
    st.rerun()

# Inicializar si no existen
for i, (_, ftype) in enumerate(FEATURES):
    key = f"f_{i}"
    if key not in st.session_state:
        st.session_state[key] = 0.0 if ftype == "continuous" else 0

st.markdown("---")

# ── Formulario ──
with st.form("main_form"):
    st.subheader("📝 Entrada de Datos")
    cols = st.columns(3)
    current_inputs = []
    
    for i, (label, ftype) in enumerate(FEATURES):
        col = cols[i % 3]
        key = f"f_{i}"
        if ftype == "continuous":
            inp = col.number_input(label, format="%.5f", key=key)
        else:
            inp = col.selectbox(label, options=[0, 1], key=key)
        current_inputs.append(inp)
    
    submitted = st.form_submit_button("🚀 Realizar Predicción")

# ── Validación y Predicción ──
if submitted:
    # Validación: En este caso Streamlit asegura que hay datos, pero podemos verificar
    # si el usuario ha dejado todo en 0 (estado inicial sin datos)
    if all(v == 0 for v in current_inputs):
        st.warning("⚠️ Por favor, complete todos los campos con datos reales del paciente antes de continuar.")
    else:
        df_input = pd.DataFrame([current_inputs], columns=[f"feature_{i+1}" for i in range(21)])
        scaled = scaler.transform(df_input)
        
        prediction = model_to_use.predict(scaled)[0]
        probs = model_to_use.predict_proba(scaled)[0] if hasattr(model_to_use, "predict_proba") else None
        
        CLASS_MAP = {1: "Hipertiroidismo", 2: "Hipotiroidismo", 3: "Normal"}
        res_text = CLASS_MAP.get(prediction, f"Clase {prediction}")
        
        st.success(f"### Resultado: {res_text}")
        st.info("💡 **Nota**: Esta predicción se basa exclusivamente en los parámetros ingresados. Verifique siempre con pruebas de laboratorio adicionales.")

st.sidebar.markdown("---")
st.sidebar.header("🏆 Desempeño General")
st.sidebar.caption("Evaluación sobre UCI Test Set")

X_gen, y_gen = load_official_test_data()
if X_gen is not None:
    X_gen.columns = [f"feature_{i+1}" for i in range(21)]
    y_gen_pred = model_to_use.predict(scaler.transform(X_gen))
    
    # Matriz en sidebar
    st.sidebar.pyplot(plot_cm(confusion_matrix(y_gen, y_gen_pred), f"Matriz General"))
    
    # Métricas detalladas debajo de la matriz
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Accuracy", f"{accuracy_score(y_gen, y_gen_pred):.1%}")
    c2.metric("F1-Score", f"{f1_score(y_gen, y_gen_pred, average='macro'):.1%}")
    
    c3, c4 = st.sidebar.columns(2)
    c3.metric("Precision", f"{precision_score(y_gen, y_gen_pred, average='macro'):.1%}")
    c4.metric("Recall", f"{recall_score(y_gen, y_gen_pred, average='macro'):.1%}")
else:
    st.sidebar.error("Archivo `ann-test.data` no encontrado")

st.sidebar.markdown("---")
st.sidebar.caption("🩺 IA Tiroidea v2.0 - Proyecto Final")

