import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  
from sklearn.ensemble import RandomForestRegressor
import os
import base64

st.set_page_config(page_title="EmpatiA Predictor", layout="wide", page_icon="🎓")

# 1. LOGIN DI SICUREZZA
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    st.subheader("🔒 Accesso Riservato Admin")
    with st.form("Login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("ACCEDI 🚀", use_container_width=True):
            if u == "admin" and p == "ia2026":
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("Credenziali errate.")
    st.stop()

# 2. LOGICA CARICAMENTO BANNER/LOGO
try:
    with open("logo.png", "rb") as img:
        b64 = base64.b64encode(img.read()).decode()
    st.markdown(f'<div style="text-align:center; background:#1E3A8A; padding:15px; border-radius:15px; margin-bottom:20px;"><img src="data:image/png;base64,{b64}" style="max-width:120px; background:white; padding:5px; border-radius:5px;"><h3 style="color:white; margin:10px 0 0 0;">Student Predictor AI</h3></div>', unsafe_allow_html=True)
except:
    st.title("🎓 Student Predictor AI")

# 3. CARICAMENTO DATI VELOCE
path = "train_leggero.csv" if os.path.exists("train_leggero.csv") else "train.csv"
df_orig = pd.read_csv(path).drop(columns=['id'], errors='ignore').drop_duplicates().fillna(0)

df_enc = df_orig.copy()
mappature = {}
for col in df_enc.select_dtypes(include=['object']).columns:
    mappature[col] = df_enc[col].astype('category').cat.categories
    df_enc[col] = df_enc[col].astype('category').cat.codes

X = df_enc.drop(columns=[df_enc.columns[-1]])
y = df_enc[df_enc.columns[-1]]

@st.cache_resource
def train_fast():
    model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model
modello_rf = train_fast()

# 4. INTERFACCIA A SCHEDE NATIVE
t1, t2 = st.tabs(["📊 Analisi Dati", "🔮 Simulatore"])

with t1:
    st.metric("Righe totali", df_orig.shape[0])
    st.metric("Colonne totali", df_orig.shape[1])
    corr = df_enc.select_dtypes(include=[np.number]).corr()
    st.plotly_chart(px.imshow(corr, text_auto='.2f', color_continuous_scale='Blugrn').update_layout(height=350), use_container_width=True)

with t2:
    st.subheader("Simulatore Predittivo")
    in_eta = st.number_input("Età Anagrafica", 15, 90, 20)
    in_genere = st.selectbox("Genere", list(mappature.get('gender', ['male', 'female'])))
    in_corso = st.selectbox("Corso Frequentato", list(mappature.get('course', ['b.tech', 'b.sc'])))
    in_ore = st.slider("Ore di Studio Giornaliere", 0.0, 16.0, 4.0, step=0.5)
    in_presenza = st.slider("Frequenza Lezioni %", 0.0, 100.0, 80.0, step=1.0)
    
    if st.button("🚀 CALCOLA PREVISIONE", use_container_width=True):
        input_dict = {'age': in_eta, 'gender': in_genere, 'course': in_corso, 'study_hours': in_ore, 'class_attendance': in_presenza}
        input_df = pd.DataFrame([input_dict]).reindex(columns=X.columns, fill_value=0)
        
        for col in input_df.columns:
            if col in mappature:
                lista = list(mappature[col])
                val = str(input_df[col].iloc[0]).strip()
                input_df[col] = lista.index(val) if val in lista else 0
                
        voto = modello_rf.predict(input_df)[0]
        st.success(f"🎯 Voto Finale Stimato: {voto:.2f} / 100")