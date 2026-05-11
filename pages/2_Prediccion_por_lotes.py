import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from utils.model_loader import load_models_and_scaler
from utils.styles import apply_custom_design

st.set_page_config(page_title="Análisis por Lotes", page_icon="📁", layout="wide")
apply_custom_design()

# ── Funciones de Carga de Datos ──
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
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako', ax=ax, cbar=False,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicción', color='#64FFDA')
    ax.set_ylabel('Realidad', color='#64FFDA')
    ax.set_title(title, color='#64FFDA', fontweight='bold')
    fig.patch.set_facecolor('#112240')
    ax.set_facecolor('#112240')
    ax.tick_params(colors='#64FFDA')
    plt.tight_layout()
    return fig

# ── Presentación ──
st.title("📁 Procesamiento y Evaluación por Lotes")
st.markdown("""
### Evaluación de Modelos a Escala
Compare el desempeño general del modelo (basado en el conjunto de prueba oficial de la UCI) contra sus propios datos cargados por lotes.
""")

scaler, lr_model, nn_model = load_models_and_scaler()
if not scaler or not lr_model or not nn_model:
    st.stop()

# ── Selección de Modelo ──
st.sidebar.header("Configuración")
model_choice = st.sidebar.radio("Seleccione el modelo a evaluar:", ("Red Neuronal", "Regresión Logística"))
model_to_use = nn_model if model_choice == "Red Neuronal" else lr_model

CLASS_LABELS = {1: "Hipertiroidismo", 2: "Hipotiroidismo", 3: "Normal"}

# ── SECCIÓN 1: DESEMPEÑO GENERAL (SIDEBAR) ──
with st.sidebar:
    st.markdown("---")
    st.header("🏆 Desempeño General")
    st.caption("Basado en el dataset de referencia UCI (ann-test.data)")
    
    X_gen, y_gen = load_official_test_data()
    if X_gen is not None:
        X_gen.columns = [f"feature_{i+1}" for i in range(21)]
        y_gen_pred = model_to_use.predict(scaler.transform(X_gen))
        
        # Métricas en sidebar
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Accuracy", f"{accuracy_score(y_gen, y_gen_pred):.1%}")
        col_m2.metric("F1-Score", f"{f1_score(y_gen, y_gen_pred, average='macro'):.1%}")
        
        col_m3, col_m4 = st.columns(2)
        col_m3.metric("Precision", f"{precision_score(y_gen, y_gen_pred, average='macro'):.1%}")
        col_m4.metric("Recall", f"{recall_score(y_gen, y_gen_pred, average='macro'):.1%}")
        
        # Matriz pequeña en sidebar
        st.pyplot(plot_cm(confusion_matrix(y_gen, y_gen_pred), f"Matriz General"))
    else:
        st.error("No se encontró `ann-test.data`")

# ── SECCIÓN 2: PROCESAMIENTO DE LOTE ACTUAL (MAIN) ──
st.header("📂 Análisis de Lote Actual")

uploaded_file = st.file_uploader("Suba su archivo de datos (22 columnas, formato .data o .csv)", type=['csv', 'data', 'txt'])

batch_x, batch_y = None, None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None) if uploaded_file.name.endswith('.data') else pd.read_csv(uploaded_file, header=None)
        if df.shape[1] >= 22:
            batch_x, batch_y = df.iloc[:, :21], df.iloc[:, 21]
    except Exception as e:
        st.error(f"Error al cargar archivo: {e}")

if "batch_data" in st.session_state and uploaded_file is None:
    batch_x, batch_y = st.session_state.batch_data

if batch_x is not None:
    st.markdown("---")
    batch_x.columns = [f"feature_{i+1}" for i in range(21)]
    y_pred = model_to_use.predict(scaler.transform(batch_x))
    
    st.header("📊 Resultados del Lote")
    col_res1, col_res2 = st.columns([1, 1.2])
    
    with col_res1:
        st.subheader("Matriz del Lote")
        st.pyplot(plot_cm(confusion_matrix(batch_y, y_pred), f"Matriz de Lote - {model_choice}"))
        
    with col_res2:
        st.subheader("Reporte detallado")
        report = classification_report(batch_y, y_pred, output_dict=True)
        df_rep = pd.DataFrame(report).transpose()
        st.dataframe(df_rep.style.format("{:.3f}").background_gradient(cmap='mako'))

    # Exportar
    res_df = batch_x.copy()
    res_df['Real'] = batch_y.values
    res_df['Pred'] = y_pred
    res_df['Texto'] = [CLASS_LABELS.get(int(p), f"Clase {p}") for p in y_pred]
    st.download_button("📥 Descargar Resultados", res_df.to_csv(index=False).encode('utf-8'), "resultados_batch.csv", "text/csv", use_container_width=True)
else:
    st.info("👋 Por favor, suba un archivo de datos o genere un lote aleatorio para comenzar el análisis.")


