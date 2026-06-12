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

# Funzione sicura per convertire il logo in stringa HTML (Base64)
def get_image_base64(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return None

# ==========================================
# 2. CSS STABILE PER UN LOOK PROFESSIONAL
# ==========================================
st.markdown("""
    <style>
        .stApp { background-color: #FFFFFF; color: #1C2B4C; font-family: 'Helvetica Neue', sans-serif; }
        #MainMenu, footer, header, div[data-testid="stHeader"] { visibility: hidden; display: none !important; }
        
        /* BANNER SUPERIORE GRADIENTE */
        .header-container { 
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); 
            padding: 22px; 
            border-radius: 0 0 20px 20px; 
            text-align: center; 
            margin-bottom: 20px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        } 
        .header-title { color: white !important; font-size: 19px !important; font-weight: 700; margin: 8px 0 0 0 !important; }
        .user-badge { display: inline-block; padding: 4px 10px; background: rgba(255,255,255,0.15); border-radius: 12px; color: white; font-size: 11px; margin-top: 5px; font-weight: bold; }
        
        /* DESIGN SEZIONI INTERNE */
        .section-title { font-size: 18px !important; font-weight: 700 !important; color: #1C2B4C; margin: 15px 0 5px 0 !important; } 
        .section-subtitle { font-size: 12px !important; color: #5A6D88; margin-bottom: 12px !important; }
        .sub-group-title { font-size: 14px !important; font-weight: 600 !important; color: #1E3A8A; margin-top: 10px !important; margin-bottom: 5px !important; border-left: 3px solid #2979FF; padding-left: 8px; }
        
        /* Card Contenitori */
        .stMetric { background-color: #F8FAFC; border-radius: 12px; border: 1px solid #E2E8F0; padding: 10px; }
        .form-card { background-color: #F8FAFC; border-radius: 12px; padding: 15px; border: 1px solid #E2E8F0; margin-bottom: 12px; }
        
        /* Bottone Calcola grande e azzurro */
        .stButton > button { 
            background-color: #2979FF !important; 
            color: white !important; 
            font-size: 15px !important; 
            font-weight: bold !important; 
            border-radius: 12px !important; 
            height: 46px !important; 
            border: none !important; 
            box-shadow: 0 4px 6px rgba(41,121,255,0.15) !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. BLOCCO DI AUTENTICAZIONE
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
        submit = st.form_submit_button("ACCEDI AL SISTEMA 🚀", use_container_width=True)
        
        if submit:
            if username == "admin" and password == "ia2026":
                st.session_state["authenticated"] = True
                st.session_state["username_loggato"] = "ADMIN"
                st.success("Accesso eseguito con successo!")
                st.rerun()
            else:
                st.error("Credenziali non corrette. Riprova.")
    st.stop()

# =
