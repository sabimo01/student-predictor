import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  
from sklearn.ensemble import RandomForestRegressor
import os
import base64

# 1. CONFIGURAZIONE PAGINA
st.set_page_config(
    page_title="EmpatiA Student Predictor", 
    layout="wide", 
    initial_sidebar_state="collapsed", 
    page_icon="🎓"
)

# Inizializzazione sessione di autenticazione
if "auth" not in st.session_state:
    st.session_state["auth"] = False

# 2. SCHERMATA DI LOGIN DI SICUREZZA
if not st.session_state["auth"]:
    st.markdown("""
        <style>
            #MainMenu, footer, header, div[data-testid="stHeader"] { visibility: hidden; display: none !important; }
            .stApp { background-color: #F8FAFC; }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("")
        st.write("")
        st.markdown("""
            <div style="background: white; padding: 30px; border-radius: 15px; box-shadow: 0px 4px 12px rgba(0,0,0,0.05); text-align: center; margin-top: 50px;">
                <h2 style="color: #1C2B4C; margin-bottom: 5px;">🔒 Accesso Riservato</h2>
                <p style="color: #64748B; font-size: 14px;">Inserisci le credenziali Admin del sistema.</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("Login System"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("ACCEDI ALLA PIATTAFORMA 🚀", use_container_width=True)
            
            if submit_login:
                if username == "admin" and password == "ia2026":
                    st.session_state["auth"] = True
                    st.rerun()
                else:
                    st.error("Credenziali non valide. Riprova.")
    st.stop()

# 3. STILE CSS PERSONALIZZATO (Colori EmpatiA: Blu scuro e Corallo)
st.markdown("""
    <style>
        #MainMenu, footer, header, div[data-testid="stHeader"] { visibility: hidden; display: none !important; }
        .stApp { background-color: #F8FAFC; color: #1C2B4C; }
        
        /* Stile personalizzato per le metriche */
        div[data-testid="stMetricValue"] {
            font-size: 32px !important;
            font-weight: bold !important;
            color: #1E3A8A !important;
        }
        
        /* Bottone Calcola */
        div.stButton > button {
            background-color: #1E3A8A !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 12px 20px !important;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #FF6F61 !important;
            box-shadow: 0px 4px 10px rgba(255, 111, 97, 0.4) !important;
        }
    </style>
""", unsafe_allow_html=True)

# 4. CARICAMENTO LOGO/HEADER BRANDIZZATO
try:
    with open("logo.png", "rb") as img_file:
        logo_b64 = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
        <div style="text-align:center; background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%); padding:25px; border-radius:15px; margin-bottom:25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <img src="data:image/png;base64,{logo_b64}" style="max-width:140px; background:white; padding:8px; border-radius:8px; margin-bottom:10px;">
            <h2 style="color:white; margin:0; font-weight: 600;">Student Predictor AI</h2>
            <p style="color:#94A3B8; margin:5px 0 0 0; font-size:14px;">Analisi e Modellazione Predittiva Carriere • <span style="color:#4ADE80;">● Piattaforma Online • ADMIN</span></p>
        </div>
    """, unsafe_allow_html=True)
except:
    st.markdown("""
        <div style="text-align:center; background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%); padding:25px; border-radius:15px; margin-bottom:25px;">
            <h2 style="color:white; margin:0;">🎓 Student Predictor AI</h2>
            <p style="color:#94A3B8; margin:5px 0 0 0;">Piattaforma Online • ADMIN</p>
        </div>
    """, unsafe_allow_html=True)

# 5. DATA LOADING & PREPROCESSING (COMPLETO DI TUTTE LE 13 COLONNE)
@st.cache_data
def load_full_dataset():
    path = "train_leggero.csv" if os.path.exists("train_leggero.csv") else "train.csv"
    df = pd.read_csv(path)
    if 'id' in df.columns: 
        df = df.drop(columns=['id'])
    df = df.drop_duplicates().fillna(0)
    
    # Pulizia nomi colonne
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Estrazione delle categorie per i menu a tendina
    col_testo = df.select_dtypes(include=['object']).columns
    opzioni = {col: list(df[col].unique()) for col in col_testo}
    
    # Dataset encodato per addestramento
    df_enc = df.copy()
    mappature = {}
    for col in col_testo:
        df_enc[col] = df_enc[col].astype('category')
        mappature[col] = list(df_enc[col].cat.categories)
        df_enc[col] = df_enc[col].cat.codes
        
    return df, df_enc, opzioni, mappature

df_orig, df_enc, opzioni_menu, mappature_cat = load_full_dataset()

# Separazione X e y (L'ultima colonna è il target es. exam_score o voto)
target_col = df_enc.columns[-1]
X = df_enc.drop(columns=[target_col])
y = df_enc[target_col]

# Addestramento modello ottimizzato in cache
@st.cache_resource
def train_global_model(X_data, y_data):
    model = RandomForestRegressor(n_estimators=60, max_depth=
