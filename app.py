import streamlit as st

st.set_page_config(
    page_title="Predicción de Tiroides",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🩺 Sistema de Predicción de Tiroides")
st.markdown("""
### Bienvenido a la plataforma de predicción
Esta aplicación utiliza modelos de Machine Learning (Regresión Logística y Redes Neuronales) para predecir si un paciente tiene hipertiroidismo, hipotiroidismo o estado normal (las 3 clases del dataset original).

**CONCEPTS > CODE:**
Como hemos discutido, los modelos **ya están pre-entrenados** de forma offline. Esto significa que la aplicación no desperdicia recursos reentrenando con cada recarga de página. Únicamente se cargan en caché y se ejecutan para realizar predicciones instantáneas.

⬅️ **Selecciona una opción en el menú lateral:**
- **Predicción Individual:** Ingresa los datos clínicos manualmente.
- **Predicción por Lotes:** Sube un archivo CSV para evaluar múltiples registros y visualizar la Matriz de Confusión.
""")
