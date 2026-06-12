import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import base64

# ==========================================
# 1. CONFIGURAZIONE DELLA PAGINA (MOBILE-FIRST)
# ==========================================
st.set_page_config(
    page_title="EmpatiA Predictor", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="🎓"
)

# Inizializzazione stabile degli stati della sessione
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username_loggato" not in st.session_state:
    st.session_state["username_loggato"] = ""

# Funzione sicura per convertire il logo in stringa HTML
def get_image_base64(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return None

# ==========================================
# 2. CSS STABILE PER SMARTPHONE E DESKTOP
# ==========================================
st.markdown("""
    <style>
        .stApp { background-color: #FFFFFF; color: #1C2B4C; font-family: 'Helvetica Neue', sans-serif; }
        #MainMenu, footer, header, div[data-testid="stHeader"] { visibility: hidden; display: none !important; }
        
        /* BANNER SUPERIORE */
        .header-container { 
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); 
            padding: 20px; 
            border-radius: 0 0 20px 20px; 
            text-align: center; 
            margin-bottom: 20px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        } 
        .header-title { color: white !important; font-size: 18px !important; font-weight: 700; margin: 8px 0 0 0 !important; }
        .user-badge { display: inline-block; padding: 4px 10px; background: rgba(255,255,255,0.15); border-radius: 12px; color: white; font-size: 11px; margin-top: 5px; font-weight: bold; }
        
        /* STRUTTURE INTERNE ED ELEMENTI GRAFICI */
        .section-title { font-size: 18px !important; font-weight: 700 !important; color: #1C2B4C; margin: 15px 0 5px 0 !important; } 
        .section-subtitle { font-size: 12px !important; color: #5A6D88; margin-bottom: 10px !important; }
        .stMetric { background-color: #F8FAFC; border-radius: 12px; border: 1px solid #E2E8F0; padding: 10px; }
        .form-card { background-color: #F8FAFC; border-radius: 12px; padding: 15px; border: 1px solid #E2E8F0; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. BLOCCO DI AUTENTICAZIONE (ADMIN)
# ==========================================
if not st.session_state["authenticated"]:
    st.markdown("""
        <div style="text-align: center; margin-top: 30px; background-color: #FFFFFF; padding: 25px; border-radius: 16px; border: 1px solid #E2E8F0;">
            <h2 style="margin: 0; color: #1C2B4C; font-size: 22px;">🔒 Accesso Riservato</h2>
            <p style="color: #5A6D88; margin: 8px 0 0 0; font-size:13px;">Inserisci le credenziali Admin del sistema.</p>
        </div>
        <br>
    """, unsafe_allow_html=True)
    
    with st.form("Login Form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
