import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  
import os
import base64
import pickle

# 1. CONFIGURAZIONE PAGINA
st.set_page_config(page_title="EmpatiA Predictor", layout="wide", initial_sidebar_state="collapsed", page_icon="🎓")

if "auth" not in st.session_state:
    st.session_state["auth"] = False

# 2. LOGIN ADMIN
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
        st.markdown("""
            <div style="background: white; padding: 30px; border-radius: 15px; box-shadow: 0px 4px 12px rgba(0,0,0,0.05); text-align: center; margin-top: 50px;">
                <h2 style="color: #1C2B4C; margin-bottom: 5px;">🔒 Accesso Riservato</h2>
                <p style="color: #64748B; font-size: 14px;">Inserisci le credenziali Admin del sistema.</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("Login Form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("ACCEDI ALLA PIATTAFORMA 🚀", use_container_width=True):
                if username == "admin" and password == "ia2026":
                    st.session_state["auth"] = True
                    st.rerun()
                else:
                    st.error("Credenziali errate.")
    st.stop()

# 3. CSS MINIMO PER IL DESIGN
st.markdown("""
    <style>
        #MainMenu, footer, header, div[data-testid="stHeader"] { visibility: hidden; display: none !important; }
        .stApp { background-color: #F8FAFC; color: #1C2B4C; }
        div[data-testid="stMetricValue"] { font-size: 32px !important; font-weight: bold !important; color: #1E3A8A !important; }
        div.stButton > button { background-color: #1E3A8A !important; color: white !important; font-weight: bold !important; border-radius: 10px !important; border: none !important; padding: 12px 20px !important; }
        div.stButton > button:hover { background-color: #FF6F61 !important; }
    </style>
""", unsafe_allow_html=True)

# 4. HEADER CON LOGO
try:
    with open("logo.png", "rb") as img:
        b64 = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <div style="text-align:center; background:linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%); padding:20px; border-radius:15px; margin-bottom:20px;">
            <img src="data:image/png;base64,{b64}" style="max-width:130px; background:white; padding:5px; border-radius:5px; margin-bottom:10px;">
            <h3 style="color:white; margin:0;">Student Predictor AI</h3>
            <p style="color:#94A3B8; margin:5px 0 0 0; font-size:12px;">Piattaforma Online • ADMIN</p>
        </div>
    """, unsafe_allow_html=True)
except:
    st.title("🎓 Student Predictor AI")

# 5. CARICAMENTO DATI VELOCE (13 COLONNE)
path = "train_leggero.csv" if os.path.exists("train_leggero.csv") else "train.csv"
df_orig = pd.read_csv(path).drop(columns=['id'], errors='ignore').drop_duplicates().fillna(0)
df_orig.columns = [c.strip().lower() for c in df_orig.columns]

df_enc = df_orig.copy()
mappature = {}
for col in df_enc.select_dtypes(include=['object']).columns:
    mappature[col] = list(df_enc[col].astype('category').cat.categories)
    df_enc[col] = df_enc[col].astype('category').cat.codes

X = df_enc.drop(columns=[df_enc.columns[-1]])

# CARICAMENTO DEL MODELLO SALVATO (O SALVAGENTE AUTOMATICO)
if os.path.exists("modello_pronto.pkl"):
    with open("modello_pronto.pkl", "rb") as f:
        modello_rf = pickle.load(f)
else:
    from sklearn.ensemble import RandomForestRegressor
    modello_rf = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1).fit(X, df_enc.iloc[:,-1])

# 6. NAVIGAZIONE SCHEDE
tab1, tab2 = st.tabs(["📊 Analisi Dati (EDA)", "🔮 Simulatore"])

with tab1:
    st.subheader("Data Profiling & Distribuzioni")
    m1, m2 = st.columns(2)
    m1.metric("Numero Totale Righe", df_orig.shape[0])
    m2.metric("Numero di Colonne", df_orig.shape[1])
    
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### 🗺️ Heatmap delle Correlazioni")
        corr = df_enc.select_dtypes(include=[np.number]).corr()
        st.plotly_chart(px.imshow(corr, text_auto='.2f', color_continuous_scale='Blugrn').update_layout(height=380, margin=dict(l=20,r=20,t=20,b=20)), use_container_width=True)
    with g2:
        st.markdown("#### 📊 Distribuzione Voto Target")
        st.plotly_chart(px.histogram(df_orig, x=df_orig.columns[-1], color_discrete_sequence=['#FF6F61']).update_layout(height=380, margin=dict(l=20,r=20,t=20,b=20)), use_container_width=True)

with tab2:
    st.subheader("Simulatore d'Impatto in Tempo Reale")
    c_left, c_mid, c_right = st.columns(3)
    
    with c_left:
        in_age = st.number_input("Età Anagrafica", 15, 90, 20)
        in_gender = st.selectbox("Genere", mappature.get('gender', ['male', 'female']))
        in_course = st.selectbox("Corso Frequentato", mappature.get('course', ['b.tech', 'b.sc', 'b.com']))
        in_method = st.selectbox("Metodo di Studio", mappature.get('study_method', ['online videos', 'textbooks']))

    with c_mid:
        in_hours = st.slider("Ore di Studio Giornaliere", 0.0, 16.0, 4.0, step=0.5)
        in_attendance = st.slider("Frequenza Lezioni %", 0.0, 100.0, 80.0, step=1.0)
        in_internet = st.selectbox("Accesso a Internet", mappature.get('internet_access', ['yes', 'no']))
        in_difficulty = st.selectbox("Difficoltà Esame", mappature.get('exam_difficulty', ['medium', 'low', 'high']))

    with c_right:
        in_sleep_h = st.slider("Ore Sonno Notturne", 3.0, 14.0, 7.0, step=0.5)
        in_sleep_q = st.selectbox("Qualità del Sonno", mappature.get('sleep_quality', ['good', 'average', 'poor']))
        in_facility = st.slider("Valutazione Struttura (1-5)", 1, 5, 3)

    if st.button("🔮 CALCOLA PREVISIONE FINALE", use_container_width=True):
        mappa_input = {
            'age': in_age, 'gender': in_gender, 'course': in_course, 'study_hours': in_hours,
            'class_attendance': in_attendance, 'internet_access': in_internet, 'sleep_hours': in
