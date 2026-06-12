import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# ==========================================
# 1. CONFIGURAZIONE DELLA PAGINA
# ==========================================
st.set_page_config(
    page_title="Student Predictor AI",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🎓"
)

# ==========================================
# 2. CSS PULITO - TITOLI PICCOLI ED ELEGANTI
# ==========================================
st.markdown("""
    <style>
        :root { 
            --primary-bg: #FFFFFF; 
            --text-dark: #1C2B4C; 
            --text-light: #5A6D88; 
            --accent: #2979FF; 
        }
        .stApp { 
            background-color: var(--primary-bg); 
            color: var(--text-dark); 
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; 
        }
        
        #MainMenu, footer, header, div[data-testid="stHeader"] { visibility: hidden; display: none !important; }
        
        .main-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
            padding: 25px;
            border-radius: 16px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        .main-header h1 { color: white !important; font-size: 24px !important; margin: 0 !important; font-weight: 700; }
        .main-header p { color: #93C5FD !important; font-size: 13px !important; margin: 5px 0 0 0 !important; }
        
        .section-title { 
            font-size: 18px !important; 
            font-weight: 700 !important; 
            color: var(--text-dark); 
            margin: 15px 0 5px 0 !important; 
        } 
        .section-subtitle {
            font-size: 13px !important;
            font-weight: 600;
            color: var(--text-light);
            margin-bottom: 15px !important;
        }
        
        .stMetric { background-color: #F8FAFC; border-radius: 12px; padding: 15px; border: 1px solid #E2E8F0; } 
        .form-card { background-color: #F8FAFC; border-radius: 12px; padding: 20px; border: 1px solid #E2E8F0; margin-bottom: 15px; }
        
        .stButton > button { 
            background-color: var(--accent) !important; 
            color: white !important; 
            font-size: 15px !important; 
            font-weight: bold !important; 
            border-radius: 12px !important; 
            height: 45px !important; 
            border: none !important; 
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SCHERMATA DI LOGIN
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
        <div style="text-align: center; margin-top: 50px; background-color: #FFFFFF; padding: 30px; border-radius: 16px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <h2 style="margin: 0; color: #1C2B4C; font-size: 24px;">🔒 Accesso Riservato</h2>
            <p style="color: #5A6D88; margin: 10px 0 0 0; font-size:14px;">Inserisci le credenziali per sbloccare l'applicazione.</p>
        </div>
        <br>
    """, unsafe_allow_html=True)
    
    with st.form("Login Form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("SBLOCCA APPLICAZIONE 🚀", use_container_width=True)
        
        if submit:
            if username == "admin" and password == "ia2026":
                st.session_state["authenticated"] = True
                st.success("Accesso eseguito!")
                st.rerun()
            else:
                st.error("Credenziali errate.")
    st.stop()

# ==========================================
# 4. CARICAMENTO E PREPARAZIONE DATI
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    nome_file = "train_leggero.csv" if os.path.exists("train_leggero.csv") else "train.csv"
    df = pd.read_csv(nome_file)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df = df.drop_duplicates()
    
    colonne_testo = df.select_dtypes(include=['object']).columns
    colonne_numeriche = df.select_dtypes(exclude=['object']).columns
    
    for col in colonne_numeriche:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
        
    if 'study_hours' in df.columns and 'sleep_hours' in df.columns:
        df['Carico_Totale_Ore'] = df['study_hours'] + df['sleep_hours']
    
    opzioni = {}
    for col in colonne_testo:
        df[col] = df[col].fillna('Sconosciuto').astype(str).str.strip()
        opzioni[col] = df[col].unique()
        
    df_encoded = df.copy()
    mappature = {}
    for col in colonne_testo:
        categorie = df_encoded[col].astype('category').cat.categories
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
        mappature[col] = categorie
        
    return df, df_encoded, opzioni, mappature

df_orig
