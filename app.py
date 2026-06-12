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

# CONFIGURAZIONE DELLA PAGINA
st.set_page_config(page_title="EmpatiA Predictor", layout="wide", initial_sidebar_state="collapsed", page_icon="🎓")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "page" not in st.session_state:
    st.session_state.page = "EDA"

# FUNZIONE PER IL LOGO
def get_image_base64(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return None

# CSS MINIMO (Solo per nascondere i menu di fabbrica)
st.markdown("""
    <style>
        #MainMenu, footer, header, div[data-testid="stHeader"] { visibility: hidden; display: none !important; }
        .stApp { background-color: #FFFFFF; color: #1C2B4C; }
    </style>
""", unsafe_allow_html=True)

# LOGIN ADMIN
if not st.session_state["authenticated"]:
    st.subheader("🔒 Accesso Riservato")
    with st.form("Login Form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("ACCEDI 🚀", use_container_width=True)
        if submit:
            if username == "admin" and password == "ia2026":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Credenziali errate.")
    st.stop()

# HEADER CON LOGO
logo_base64 = get_image_base64("logo.png")
if logo_base64:
    st.markdown(f'<div style="text-align:center; background:#1E3A8A; padding:20px; border-radius:15px; margin-bottom:20px;"><img src="data:image/png;base64,{logo_base64}" style="max-width:130px; background:white; padding:5px; border-radius:5px;"><h3 style="color:white; margin:10px 0 0 0;">Student Predictor AI</h3></div>', unsafe_allow_html=True)
else:
    st.title("🎓 Student Predictor AI")

# NAVIGAZIONE CON SCHEDE NATIVE
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔮 Simulatore", "📈 Performance"])

# CARICAMENTO DATI (13 COLONNE)
@st.cache_data
def load_and_preprocess():
    path = "train_leggero.csv" if os.path.exists("train_leggero.csv") else "train.csv"
    df = pd.read_csv(path)
    if 'id' in df.columns: df = df.drop(columns=['id'])
    df = df.drop_duplicates().fillna(0)
    col_testo = df.select_dtypes(include=['object']).columns
    opzioni = {col: df[col].unique() for col in col_testo}
    df_enc = df.copy()
    mappature = {}
    for col in col_testo:
        df_enc[col] = df_enc[col].astype('category').cat.codes
        mappature[col] = df[col].astype('category').cat.categories
    return df, df_enc, opzioni, mappature

df_orig, df_enc, opzioni_menu, mappature_cat = load_and_preprocess()
target_col = df_enc.columns[-1]
X = df_enc.drop(columns=[target_col])
y = df_enc[target_col]
X_train, X_test, y_train, y_test =
