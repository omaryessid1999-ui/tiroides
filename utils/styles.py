import streamlit as st

def apply_custom_design():
    """
    Injects custom CSS to create a professional medical dashboard aesthetic.
    Uses glassmorphism, modern typography, and a clinical color palette.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="st-"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0A192F 0%, #112240 100%);
        color: #E6F1FF;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background-color: rgba(17, 34, 64, 0.7) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(100, 255, 218, 0.1);
    }

    /* Titles and Headers */
    h1, h2, h3 {
        color: #64FFDA !important;
        font-weight: 700 !important;
    }

    /* Custom Cards for Containers */
    .st-emotion-cache-12w0qpk, .st-emotion-cache-1r6slb0 {
        background: rgba(23, 42, 69, 0.6) !important;
        padding: 2rem !important;
        border-radius: 15px !important;
        border: 1px solid rgba(100, 255, 218, 0.1) !important;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    /* Buttons */
    .stButton > button {
        background-color: transparent !important;
        color: #64FFDA !important;
        border: 1px solid #64FFDA !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        background-color: rgba(100, 255, 218, 0.1) !important;
        box-shadow: 0 0 15px rgba(100, 255, 218, 0.2) !important;
        transform: translateY(-2px);
    }

    /* Inputs */
    .stTextInput > div > div > input, .stNumberInput > div > div > input {
        background-color: #0A192F !important;
        color: white !important;
        border: 1px solid rgba(100, 255, 218, 0.2) !important;
        border-radius: 8px !important;
    }

    /* Success/Info Boxes */
    .stAlert {
        background-color: rgba(100, 255, 218, 0.05) !important;
        color: #64FFDA !important;
        border: 1px solid rgba(100, 255, 218, 0.2) !important;
        border-radius: 10px !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

def landing_page_hero():
    """Displays the hero section for the landing page."""
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("Inteligencia Médica para el Análisis Tiroideo")
        st.markdown("""
        Utilizamos arquitecturas de **Redes Neuronales Profundas** y **Regresión Logística** 
        para proporcionar diagnósticos preliminares con precisión clínica.
        
        Nuestra plataforma procesa parámetros complejos para identificar patrones 
        de hipertiroidismo e hipotiroidismo de forma instantánea.
        """)
        if st.button("Comenzar Diagnóstico"):
            st.info("Utilice el menú lateral para navegar a 'Predicción Individual'.")
            
    with col2:
        st.image("assets/hero.png", use_container_width=True)
