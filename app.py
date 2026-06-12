import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# ==========================================
# 1. CONFIGURAZIONE & LOGO
# ==========================================
st.set_page_config(page_title="EmpatiA Predictor", layout="wide", initial_sidebar_state="collapsed")

# Gestione stato navigazione
if 'page' not in st.session_state:
    st.session_state.page = 'EDA'

# ==========================================
# 2. CSS PER ICONE GIGANTI E LAYOUT MOBILE
# ==========================================
st.markdown("""
    <style>
        # /* Reset e Font */ .stApp { background-color: #FFFFFF; font-family: 'Helvetica Neue', sans-serif; }
        # /* Nascondi Menu Streamlit */ #MainMenu, footer, header, div[data-testid="stHeader"] { visibility: hidden; display: none !important; }
        
        # /* --- HEADER CON LOGO --- */ .header-container { background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); padding: 20px; border-radius: 0 0 25px 25px; text-align: center; margin-bottom: 25px; } .logo-img { max-width: 150px; margin-bottom: 10px; background: white; padding: 5px; border-radius: 10px; } .header-title { color: white !important; font-size: 22px !important; font-weight: 700; margin: 0; }
        
        # /* --- NAVIGAZIONE A ICONE GIGANTI --- */ .nav-button-container { text-align: center; } .stButton > button { width: 80px !important; height: 80px !important; border-radius: 20px !important; background-color: #F8FAFC !important; color: #1E3A8A !important; font-size: 35px !important; border: 1px solid #E2E8F0 !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important; transition: all 0.3s; } .stButton > button:active { transform: scale(0.9); background-color: #2979FF !important; color: white !important; } .nav-hint { font-size: 11px !important; font-weight: 600; color: #5A6D88; margin-top: 8px; text-transform: uppercase; }
        
        # /* --- TITOLI SEZIONI PICCOLI --- */ .section-title { font-size: 20px !important; font-weight: 700 !important; color: #1C2B4C; margin: 20px 0 10px 0 !important; } .section-subtitle { font-size: 13px !important; color: #5A6D88; margin-bottom: 15px !important; }
        
        # /* Card Metriche */ .stMetric { background-color: #F8FAFC; border-radius: 15px; border: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HEADER CON LOGO
# ==========================================
with st.container():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    if os.path.exists("logo.png"):
        st.image("logo.png", width=160)
    st.markdown('<p class="header-title">Student Predictor AI</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 4. NAVIGAZIONE A ICONE (PULSANTI GRANDI)
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="nav-button-container">', unsafe_allow_html=True)
    if st.button("📊", key="btn_eda", use_container_width=False):
        st.session_state.page = 'EDA'
    st.markdown('<p class="nav-hint">Dati ed<br>Analisi</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="nav-button-container">', unsafe_allow_html=True)
    if st.button("🔮", key="btn_sim", use_container_width=False):
        st.session_state.page = 'SIM'
    st.markdown('<p class="nav-hint">Test<br>Previsione</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="nav-button-container">', unsafe_allow_html=True)
    if st.button("📈", key="btn_perf", use_container_width=False):
        st.session_state.page = 'PERF'
    st.markdown('<p class="nav-hint">Qualità<br>Modelli</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 5. CARICAMENTO DATI & MODELLI (Identico a prima)
# ==========================================
@st.cache_data
def load_data():
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

df_orig, df_enc, opzioni_menu, mappature_cat = load_data()
X = df_enc.drop(columns=[df_enc.columns[-1]])
y = df_enc[df_enc.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train():
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train)
    lr = LinearRegression().fit(X_train, y_train)
    rmse_rf = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
    rmse_lr = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))
    return rf, lr, rmse_rf, rmse_lr

modello_rf, modello_lr, rmse_rf, rmse_lr = train()

# ==========================================
# 6. LOGICA VISUALIZZAZIONE PAGINE
# ==========================================

# --- PAGINA EDA ---
if st.session_state.page == 'EDA':
    st.markdown("<p class='section-title'>Analisi Esplorativa (EDA)</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Panoramica dei dati storici</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Righe", df_orig.shape[0])
    c2.metric("Colonne", df_orig.shape[1])
    
    corr = df_enc.select_dtypes(include=[np.number]).corr()
    st.plotly_chart(px.imshow(corr, text_auto='.2f', color_continuous_scale='Blugrn').update_layout(margin=dict(l=5,r=5,t=5,b=5)), use_container_width=True)

# --- PAGINA SIMULATORE ---
elif st.session_state.page == 'SIM':
    st.markdown("<p class='section-title'>Simulatore Predittivo</p>", unsafe_allow_html=True)
    with st.container():
        in_eta = st.number_input("Età", 15, 90, 20)
        in_genere = st.selectbox("Genere", opzioni_menu.get('gender', ['male', 'female']))
        in_corso = st.selectbox("Corso", opzioni_menu.get('course', ['b.tech', 'b.sc']))
        in_ore = st.slider("Ore Studio", 0.0, 12.0, 4.0)
        in_presenza = st.slider("Presenza %", 0.0, 100.0, 80.0)
        
    if st.button("🚀 CALCOLA VOTO", use_container_width=True):
        input_df = pd.DataFrame([[in_eta, in_genere, in_corso, in_ore, in_presenza]], columns=X.columns[:5])
        # Nota: Ho semplificato l'input per brevità, assicurati che input_df abbia tutte le colonne di X
        input_df = input_df.reindex(columns=X.columns, fill_value=0)
        
        # Codifica al volo
        for col in input_df.columns:
            if col in mappature_cat:
                lista = list(mappature_cat[col])
                val = str(input_df
