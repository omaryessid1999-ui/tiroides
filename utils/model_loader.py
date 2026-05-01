import streamlit as st
import joblib
import os

@st.cache_resource
def load_models_and_scaler():
    """
    Carga los modelos y el escalador solo una vez gracias a st.cache_resource.
    Esto soluciona el antipatrón de rendimiento de reentrenar en cada recarga.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    try:
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        lr_model = joblib.load(os.path.join(models_dir, 'logistic_regression.pkl'))
        nn_model = joblib.load(os.path.join(models_dir, 'neural_network.pkl'))
        return scaler, lr_model, nn_model
    except Exception as e:
        st.error(f"Error al cargar los modelos. ¿Ejecutaste train_models.py primero? Detalles: {e}")
        return None, None, None
