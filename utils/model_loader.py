import streamlit as st
import os
import joblib

@st.cache_resource(show_spinner="Cargando modelos de diagnóstico...")
def load_models_and_scaler():
    """
    Carga los modelos pre-entrenados desde disco.
    Usa @st.cache_resource para que solo se ejecute una vez por sesión del servidor.
    Si los archivos no existen, indica al usuario que ejecute train_models.py.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    required = ["scaler.pkl", "logistic_regression.pkl", "neural_network.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(models_dir, f))]

    if missing:
        st.error(
            f"⚠️ Modelos no encontrados: `{', '.join(missing)}`.\n\n"
            "Ejecuta `python train_models.py` en la terminal para generarlos."
        )
        return None, None, None

    try:
        scaler   = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        lr_model = joblib.load(os.path.join(models_dir, "logistic_regression.pkl"))
        nn_model = joblib.load(os.path.join(models_dir, "neural_network.pkl"))
        return scaler, lr_model, nn_model
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None, None, None
