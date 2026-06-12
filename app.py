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
# 1. CONFIGURAZIONE DELLA PAGINA
# ==========================================
st.set_page_config(
    page_title="EmpatiA Predictor", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="🎓"
)

# Gestione dello stato della navigazione mobile
if 'page' not in st.session_state:
    st.session_state.page = 'EDA'

# Funzione per convertire il logo in formato sicuro per l'HTML
def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# ==========================================
# 2. CSS AVANZATO PER IL LAYOUT MOBILE
# ==========================================
st.markdown("""
    <style>
        .stApp { background-color: #FFFFFF; font-family: 'Helvetica Neue', sans-serif; }
        
        /* Nasconde i menu nativi di Streamlit */
        #MainMenu, footer, header, div[data-testid="stHeader"] { visibility: hidden; display: none !important; }
        
        /* HEADER BOX PULITO CON GRADIENTE */
        .header-container { 
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); 
            padding: 20px; 
            border-radius: 0 0 25px 25px; 
            text-align: center; 
            margin-bottom: 25px; 
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        } 
        .header-title { color: white !important; font-size: 20px !important; font-weight: 700; margin: 8px 0 0 0 !important; }
        
        /* FORZA LE COLONNE A RIMANERE AFFIANCATE SU MOBILE */
        div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            justify-content: space-between !important;
            gap: 10px !important;
        }
        div[data-testid="column"] {
            min-width: auto !important;
            flex: 1 !important;
            text-align: center !important;
        }
        
        /* STILE ICONE DI NAVIGAZIONE GRANDI */
        .nav-button-container { text-align: center; width: 100%; } 
        .stButton > button { 
            width: 75px !important; 
            height: 75px !important; 
            border-radius: 18px !important; 
            background-color: #F8FAFC !important; 
            color: #1E3A8A !important; 
            font-size: 30px !important; 
            border: 1px solid #E2E8F0 !important; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.04) !important;
            margin: 0 auto !important;
            display: block !important;
        } 
        .nav-hint { font-size: 11px !important; font-weight: 700; color: #5A6D88; margin-top: 6px; text-transform: uppercase; line-height: 1.2; }
        
        /* TITOLI SEZIONI PICCOLI ED ELEGANTI (Richiesta 3) */
        .section-title { font-size: 18px !important; font-weight: 700 !important; color: #1C2B4C; margin: 15px 0 5px 0 !important; } 
        .section-subtitle { font-size: 13px !important; color: #5A6D88; margin-bottom: 15px !important; }
        
        /* Struttura Card */
        .stMetric { background-color: #F8FAFC; border-radius: 14px; border: 1px solid #E2E8F0; padding: 12px; }
        .form-card { background-color: #F8FAFC; border-radius: 14px; padding: 18px; border: 1px solid #E2E8F0; margin-bottom: 12px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. GENERAZIONE HEADER INTEGRATO (LOGO + TITOLO)
# ==========================================
if os.path.exists("logo.png"):
    base64_logo = get_image_base64("logo.png")
    html_header = f"""
    <div class="header-container">
        <img src="data:image/png;base64,{base64_logo}" style="max-width: 140px; background: white; padding: 6px; border-radius: 8px;">
        <p class="header-title">Student Predictor AI</p>
    </div>
    """
else:
    html_header = """
    <div class="header-container">
        <p class="header-title">Student Predictor AI</p>
    </div>
    """
st.markdown(html_header, unsafe_allow_html=True)

# ==========================================
# 4. NAVIGAZIONE ORIZZONTALE A ICONE
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="nav-button-container">', unsafe_allow_html=True)
    if st.button("📊", key="btn_eda"):
        st.session_state.page = 'EDA'
    st.markdown('<p class="nav-hint">Dati ed<br>Analisi</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="nav-button-container">', unsafe_allow_html=True)
    if st.button("🔮", key="btn_sim"):
        st.session_state.page = 'SIM'
    st.markdown('<p class="nav-hint">Test<br>Previsione</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="nav-button-container">', unsafe_allow_html=True)
    if st.button("📈", key="btn_perf"):
        st.session_state.page = 'PERF'
    st.markdown('<p class="nav-hint">Qualità<br>Modelli</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 5. CODICE DI ELABORAZIONE DATI E MODELLI
# ==========================================
@st.cache_data
def load_data():
    path = "train_leggero.csv" if os.path.exists("train_leggero.csv") else "train.csv"
    df = pd.read_csv(path)
    if 'id' in df.columns: 
        df = df.drop(columns=['id'])
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
# 6. LOGICA DI CAMBIO SCHEDA
# ==========================================

# --- SCHEDA 1: EDA ---
if st.session_state.page == 'EDA':
    st.markdown("<p class='section-title'>Analisi Esplorativa (EDA)</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Panoramica della base dati</p>", unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Righe totali", df_orig.shape[0])
    with m2:
        st.metric("Colonne totali", df_orig.shape[1])
    
    st.markdown("<p class='section-title'>📊 Grafico delle Correlazioni</p>", unsafe_allow_html=True)
    corr = df_enc.select_dtypes(include=[np.number]).corr()
    st.plotly_chart(px.imshow(corr, text_auto='.2f', color_continuous_scale='Blugrn').update_layout(margin=dict(l=5,r=5,t=5,b=5)), use_container_width=True)

# --- SCHEDA 2: SIMULATORE ---
elif st.session_state.page == 'SIM':
    st.markdown("<p class='section-title'>Simulatore Predittivo</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    in_eta = st.number_input("Età Anagrafica", 15, 90, 20)
    in_genere = st.selectbox("Genere", opzioni_menu.get('gender', ['male', 'female']))
    in_corso = st.selectbox("Corso di Studi", opzioni_menu.get('course', ['b.tech', 'b.sc']))
    in_ore = st.slider("Ore Studio Giornaliere", 0.0, 12.0, 4.0, step=0.5)
    in_presenza = st.slider("Presenza alle Lezioni %", 0.0, 100.0, 80.0, step=1.0)
    st.markdown('</div>', unsafe_allow_html=True)
        
    if st.button("🚀 CALCOLA PREVISIONE VOTO", use_container_width=True):
        input_df = pd.DataFrame([[in_eta, in_genere, in_corso, in_ore, in_presenza]], columns=X.columns[:5])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)
        
        for col in input_df.columns:
            if col in mappature_cat:
                lista = list(mappature_cat[col])
                val = str(input_df[col].iloc[0])
                input_df[col] = lista.index(val) if val in lista else 0
        
        voto = modello_rf.predict(input_df)[0]
        st.markdown(f"""
            <div style="background-color: #F1F5F9; padding: 15px; border-radius: 12px; text-align: center; border: 1px solid #CBD5E1; margin-top: 15px;">
                <p style="color: #5A6D88; margin:0; font-size:13px;">Esito Proiezione Voto</p>
                <p style="color: #1E3A8A; font-size: 32px; font-weight: bold; margin:5px 0 0 0;">{voto:.2f} / 100</p>
            </div>
        """, unsafe_allow_html=True)

# --- SCHEDA 3: PERFORMANCE ---
elif st.session_state.page == 'PERF':
    st.markdown("<p class='section-title'>Affidabilità Algoritmi</p>", unsafe_allow_html=True)
    
    p1, p2 = st.columns(2)
    with p1:
        st.metric("Errore RF", f"{rmse_rf:.4f}")
    with p2:
        st.metric("Errore LR", f"{rmse_lr:.4f}")
    st.info("I punteggi mostrano lo scarto medio del modello. Valori minori indicano stime più precise.")

st.write("---")
with st.expander("Ispeziona un'anteprima dei dati storici"):
    st.dataframe(df_orig.head(5), use_container_width=True)
