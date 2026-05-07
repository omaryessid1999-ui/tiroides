import streamlit as st
from utils.styles import apply_custom_design, landing_page_hero

st.set_page_config(
    page_title="Thyroid AI - Diagnóstico Predictivo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Aplicar diseño profesional
apply_custom_design()

# Mostrar Hero Section
landing_page_hero()

st.markdown("---")

st.subheader("💡 Fundamentos del Sistema")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🧠 Modelos Pre-entrenados")
    st.write("Optimizamos el rendimiento cargando modelos serializados, evitando el sobrecosto de entrenamiento en tiempo de ejecución.")

with col2:
    st.markdown("### 📊 Análisis Multiclase")
    st.write("Clasificación precisa entre Hipertiroidismo, Hipotiroidismo y estado Normal basada en 21 biomarcadores clínicos.")

with col3:
    st.markdown("### 🚀 Escalabilidad")
    st.write("Capacidad de procesamiento por lotes para análisis masivos de datos clínicos con visualización de métricas de desempeño.")
