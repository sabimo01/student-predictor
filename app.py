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
# 1. AUTENTICAZIONE E AUTORIZZAZIONE
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username_loggato" not in st.session_state:
    st.session_state["username_loggato"] = ""

if not st.session_state["authenticated"]:
    st.set_page_config(page_title="🔒 Login - Student Predictor AI", layout="centered")
    
    st.markdown("""
        <div style="text-align: center; margin-top: 50px; background-color: #f0f2f6; padding: 25px; border-radius: 10px; border: 1px solid #e0e0e0;">
            <h2 style="margin: 0; color: #1572B6;">🔒 Accesso Riservato Piattaforma</h2>
            <p style="color: #555; margin: 10px 0 0 0;">Inserisci le credenziali autorizzate per sbloccare la dashboard e la Knowledge Base.</p>
        </div>
        <br>
    """, unsafe_allow_html=True)
    
    with st.form("Login Form"):
        username = st.text_input("Username Autorizzato")
        password = st.text_input("Password di Sicurezza", type="password")
        submit = st.form_submit_button("SBLOCCA DASHBOARD 🚀", use_container_width=True)
        
        if submit:
            if username == "admin" and password == "ia2026":
                st.session_state["authenticated"] = True
                st.session_state["username_loggato"] = username  
                st.success("Accesso eseguito! Caricamento in corso...")
                st.rerun()
            else:
                st.error("Credenziali non corrette o utente non autorizzato.")
    st.stop()

# ==========================================
# 2. CONFIGURAZIONE PAGINA E INTERFACCIA (POST-LOGIN)
# ==========================================
st.set_page_config(page_title="Student Predictor AI - Piattaforma Pro", layout="wide")

col_logo, col_titolo = st.columns([1, 5])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
    else:
        st.markdown("<h1 style='font-size: 4rem; margin:0;'>🎓</h1>", unsafe_allow_html=True)
with col_titolo:
    st.markdown(f"""
        <h1 style='margin:0; color:#1572B6;'>Student Predictor AI Dashboard</h1>
        <div style='display: flex; gap: 15px; align-items: center; margin-top: 5px;'>
            <p style='color:#666; margin:0; font-size:1.1rem;'>Sistema di monitoraggio delle carriere e modellazione predittiva</p>
            <span style='background-color: #e1f5fe; color: #0288d1; padding: 3px 10px; border-radius: 15px; font-size: 0.85rem; font-weight: bold; border: 1px solid #b3e5fc;'>
                🟢 Utente: {st.session_state["username_loggato"]}
            </span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 3. PIPELINE DI PREPROCESSING & FEATURE ENGINEERING
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    nome_file_csv = "train_leggero.csv" if os.path.exists("train_leggero.csv") else "train.csv"
    df = pd.read_csv(nome_file_csv)
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    df = df.drop_duplicates()
    
    colonne_testo = df.select_dtypes(include=['object']).columns
    colonne_numeriche = df.select_dtypes(exclude=['object']).columns
    
    for col in colonne_numeriche:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
        
    if 'study_hours' in df.columns:
        df['Carico_Totale_Ore'] = df['study_hours'] + df.get('sleep_hours', 7.0)
    
    mappature_opzioni = {}
    for col in colonne_testo:
        df[col] = df[col].fillna('Sconosciuto').astype(str).str.strip()
        mappature_opzioni[col] = df[col].unique()
        
    df_encoded = df.copy()
    codici_categorie = {}
    for col in colonne_testo:
        categorie = df_encoded[col].astype('category').cat.categories
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
        codici_categorie[col] = categorie
        
    return df, df_encoded, mappature_opzioni, codici_categorie

df_originale, df_elaborato, opzioni_menu, codici_categorie = load_and_preprocess_data()
target_col = df_elaborato.columns[-2] 

X = df_elaborato.drop(columns=[target_col])
y = df_elaborato[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_models(X_t, X_v, y_t, y_v):
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_t, y_t)
    pred_rf = rf.predict(X_v)
    rmse_rf = np.sqrt(mean_squared_error(y_v, pred_rf))
    
    lr = LinearRegression()
    lr.fit(X_t, y_t)
    pred_lr = lr.predict(X_v)
    rmse_lr = np.sqrt(mean_squared_error(y_v, pred_lr))
    
    return rf, lr, rmse_rf, rmse_lr

modello_rf, modello_lr, rmse_rf, rmse_lr = train_models(X_train, X_test, y_train, y_test)

# ==========================================
# 4. CREAZIONE STRUTTURA A SCHEDE (TABS)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis (EDA)", "🔮 Predictor Dashboard", "📈 Performance Modelli"])

# ------------------------------------------
# SCHEDA 1: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------
with tab1:
    st.markdown("<h2 style='font-size: 22px; font-weight: bold; margin-top: 0px; margin-bottom: 15px;'>Analisi Esplorativa dei Dati (EDA)</h2>
