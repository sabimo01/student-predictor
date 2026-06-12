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
        
        /* SCELTA DESIGN SEZIONI INTERNE */
        .section-title { font-size: 18px !important; font-weight: 700 !important; color: #1C2B4C; margin: 15px 0 5px 0 !important; } 
        .section-subtitle { font-size: 12px !important; color: #5A6D88; margin-bottom: 12px !important; }
        
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
# 3. BLOCCO DI AUTENTICAZIONE CORRETTO (CON SUBMIT FORM)
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
        # LA CORREZIONE: st.form_submit_button risolve il bug "Missing Submit Button"
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

# ==========================================
# 4. HEADER DI TESTATA CON LOGO EMPATIA
# ==========================================
logo_base64 = get_image_base64("logo.png")
if logo_base64:
    html_header = f"""
    <div class="header-container">
        <img src="data:image/png;base64,{logo_base64}" style="max-width: 120px; background: white; padding: 4px; border-radius: 6px;">
        <p class="header-title">Student Predictor AI</p>
        <span class="user-badge">🟢 UTENTE: {st.session_state["username_loggato"]}</span>
    </div>
    """
else:
    html_header = f"""
    <div class="header-container">
        <p class="header-title">Student Predictor AI</p>
        <span class="user-badge">🟢 UTENTE: {st.session_state["username_loggato"]}</span>
    </div>
    """
st.markdown(html_header, unsafe_allow_html=True)

# ==========================================
# 5. PIPELINE DATI & MACHINE LEARNING
# ==========================================
@st.cache_data
def load_and_preprocess():
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

df_orig, df_enc, opzioni_menu, mappature_cat = load_and_preprocess()
target_col = df_enc.columns[-1]

X = df_enc.drop(columns=[target_col])
y = df_enc[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train():
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train)
    lr = LinearRegression().fit(X_train, y_train)
    rmse_rf = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
    rmse_lr = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))
    return rf, lr, rmse_rf, rmse_lr

modello_rf, modello_lr, score_rf, score_lr = train()

# ==========================================
# 6. NAVIGAZIONE STABILE CON GRAPHIC TABS EMOJI
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔮 Simulatore", "📈 Performance"])

# --- SCHEDA 1: EDA ---
with tab1:
    st.markdown("<p class='section-title'>Analisi Esplorativa (EDA)</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Panoramica dei dati caricati nel sistema</p>", unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Righe totali", df_orig.shape[0])
    with m2:
        st.metric("Colonne totali", df_orig.shape[1])
    
    st.markdown("<p class='section-title'>📊 Mappa delle Correlazioni</p>", unsafe_allow_html=True)
    corr = df_enc.select_dtypes(include=[np.number]).corr()
    fig_heat = px.imshow(corr, text_auto='.2f', color_continuous_scale='Blugrn')
    fig_heat.update_layout(margin=dict(l=5,r=5,t=5,b=5), height=350)
    st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})

# --- SCHEDA 2: SIMULATORE COMPLETO ---
with tab2:
    st.markdown("<p class='section-title'>Simulatore Predittivo Real-Time</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    in_eta = st.number_input("Età Anagrafica", 15, 90, 20)
    in_genere = st.selectbox("Genere", opzioni_menu.get('gender', ['male', 'female']))
    in_corso = st.selectbox("Course / Corso Frequentato", opzioni_menu.get('course', ['b.tech', 'b.sc', 'b.com']))
    in_ore = st.slider("Ore di Studio Giorni", 0.0, 12.0, 4.0, step=0.5)
    in_presenza = st.slider("Frequenza Lezioni %", 0.0, 100.0, 80.0, step=1.0)
    st.markdown('</div>', unsafe_allow_html=True)
        
    if st.button("🚀 CALCOLA PREVISIONE FINALE", use_container_width=True):
        input_dict = {
            'age': in_eta, 'gender': in_genere, 'course': in_corso,
            'study_hours': in_ore, 'class_attendance': in_presenza
        }
        
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)
        
        for col in input_df.columns:
            if col in mappature_cat:
                lista = list(mappature_cat[col])
                val = str(input_df[col].iloc[0])
                input_df[col] = lista.index(val) if val in lista else 0
        
        voto = modello_rf.predict(input_df)[0]
        st.markdown(f"""
            <div style="background-color: #F1F5F9; padding: 18px; border-radius: 12px; text-align: center; border: 1px solid #CBD5E1; margin-top: 10px;">
                <p style="color: #5A6D88; margin:0; font-size:12px;">Voto Finale Stimato dall'AI</p>
                <p style="color: #1E3A8A; font-size: 30px; font-weight: bold; margin:3px 0 0 0;">{voto:.2f} / 100</p>
            </div>
        """, unsafe_allow_html=True)

# --- SCHEDA 3: PERFORMANCE CORRETTE ---
with tab3:
    st.markdown("<p class='section-title'>Performance dei Modelli</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Valutazione dello scarto medio degli algoritmi</p>", unsafe_allow_html=True)
    
    p1, p2 = st.columns(2)
    with p1:
        st.metric("Errore Random Forest (RMSE)", f"{score_rf:.4f}")
    with p2:
        st.metric("Errore Linear Reg. (RMSE)", f"{score_lr:.4f}")
        
    st.info("💡 Nota informativa: I valori indicano lo scarto quadratico medio (RMSE). Più il numero si avvicina a 0, maggiore è la precisione delle stime.")

st.write("---")
with st.expander("Ispeziona un'anteprima dei dati storici"):
    st.dataframe(df_orig.head(5), use_container_width=True)
