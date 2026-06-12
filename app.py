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
# 1. CONFIGURAZIONE UNICA DELLA PAGINA - MOBILE-FIRST
# ==========================================
st.set_page_config(
    page_title="Student Predictor AI - Piattaforma Pro",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🎓"
)

# ==========================================
# 2. CSS PERSONALIZZATO PER UNA "FIGATA" GRAFICA MOBILE
# ==========================================
st.markdown("""
    <style>
        :root { --primary: #2979FF; --secondary: #1C2B4C; --accent: #FF5252; --bg-main: #F7F9FC; --bg-card: #FFFFFF; --text: #1C2B4C; }
        .stApp { background-color: var(--bg-main); color: var(--text); font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
        #MainMenu, footer, header { visibility: hidden; }
        .mobile-header-text { margin-top: -60px; }
        .main-title { font-size: 30px; font-weight: bold; color: var(--secondary); margin-bottom: 5px; margin-top: 0px;} 
        .main-subtitle { font-size: 15px; color: rgba(28,43,76,0.7); margin-bottom: 20px; margin-top: 0px;}
        .data-profiling-title { font-size: 18px; font-weight: bold; color: var(--secondary); margin-bottom: 10px; margin-top: 15px;} 
        .stMetric { background-color: var(--bg-card); border-radius: 15px; padding: 15px; border: 1px solid rgba(0,0,0,0.03); margin-bottom: 10px; text-align: left;} 
        .stMetric > div { color: rgba(28,43,76,0.5); font-size: 13px !important;} 
        .stMetric > div:first-child + div { color: var(--secondary); font-size: 32px !important; font-weight: bold !important;}
        .stDataFrame { background-color: var(--bg-card); border-radius: 15px; padding: 5px; border: 1px solid rgba(0,0,0,0.03); margin-top: 10px;}
        .stSelectbox > label, .stSlider > label, .stNumberInput > label { color: rgba(28,43,76,0.7); font-size: 14px; margin-bottom: 0px;}
        .stSlider > div { margin-top: -10px;} 
        .stSlider input[type=range] { color: var(--primary) !important; height: 10px;}
        .stButton > button { background-color: var(--accent) !important; color: white !important; font-size: 18px !important; font-weight: bold !important; border-radius: 25px !important; height: 50px !important; border: none !important; box-shadow: 0 4px 10px rgba(255,82,82,0.3) !important; transition: transform 0.1s;} 
        .stButton > button:active { transform: scale(0.98);}
        div.stTabs > div > div > button { font-size: 14px !important; color: rgba(28,43,76,0.5) !important; border-radius: 20px 20px 0 0 !important; background-color: transparent !important; border: none !important;} 
        div.stTabs > div > div > button[aria-selected="true"] { color: var(--primary) !important; font-weight: bold !important;} 
        div.stTabs > div > div > button:hover { background-color: rgba(41,121,255,0.03) !important;}
        .st-cx { width: 100px; padding: 8px; border-radius: 12px; font-size: 12px; background-color: rgba(41,121,255,0.05); color: var(--primary); border: 1px solid rgba(41,121,255,0.15);}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. AUTENTICAZIONE E AUTORIZZAZIONE
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username_loggato" not in st.session_state:
    st.session_state["username_loggato"] = ""

if not st.session_state["authenticated"]:
    st.markdown("""
        <div style="text-align: center; margin-top: 30px; background-color: #FFFFFF; padding: 30px; border-radius: 20px; border: 1px solid rgba(0,0,0,0.03); box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <h2 style="margin: 0; color: #1C2B4C; font-size: 26px;">🔒 Accesso Pro</h2>
            <p style="color: rgba(28,43,76,0.7); margin: 10px 0 0 0; font-size:14px;">Utilizza le credenziali autorizzate.</p>
        </div>
        <br>
    """, unsafe_allow_html=True)
    
    with st.form("Login Form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("SBLOCCA 🚀", use_container_width=True)
        
        if submit:
            if username == "admin" and password == "ia2026":
                st.session_state["authenticated"] = True
                st.session_state["username_loggato"] = username  
                st.success("Accesso eseguito!")
                st.rerun()
            else:
                st.error("Credenziali non corrette.")
    st.stop()

# ==========================================
# 4. INTERFACCIA HEADER MOBILE (POST-LOGIN)
# ==========================================
header_img_col = st.columns([1])[0]
with header_img_col:
    if os.path.exists("header_mobile.png"):
        st.image("header_mobile.png", use_container_width=True)
    else:
        st.markdown("""
            <div style="width: 100%; height: 140px; background: linear-gradient(135deg, #2979FF 0%, #1C2B4C 100%); border-radius: 0 0 25px 25px;"></div>
        """, unsafe_allow_html=True)

header_text_col = st.columns([1])[0]
with header_text_col:
    col_t1, col_u1 = st.columns([4, 1])
    with col_t1:
        st.markdown("""
            <div class="mobile-header-text">
                <h1 class="main-title">Student Predictor AI</h1>
                <p class="main-subtitle">Analisi e Modellazione Predittiva Carriere</p>
            </div>
        """, unsafe_allow_html=True)
    with col_u1:
        st.write(f'<div class="st-cx">Utente: {st.session_state["username_loggato"]}</div>', unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 5. PIPELINE DI PREPROCESSING & FEATURE ENGINEERING
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
# 6. CREAZIONE STRUTTURA TABS MOBILE SUPER PULITI
# ==========================================
tab1, tab2, tab3 = st.tabs(["● EDA", "● Simulatore", "● Performance"])

# ------------------------------------------
# SCHEDA 1: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------
with tab1:
    st.markdown("<h2 class='data-profiling-title'>Analisi Esplorativa (EDA)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:16px; font-weight:bold; color:#1C2B4C; margin-bottom:10px;'>Data Profiling</p>", unsafe_allow_html=True)
    
    with st.container():
        st.metric("Numero Totale Righe", df_originale.shape[0])
        st.metric("Numero di Colonne", df_originale.shape[1])
        st.write("**Tipi di Dati Rilevati:**")
        st.dataframe(df_originale.dtypes.astype(str).to_frame(name="Tipo"), use_container_width=True)
        
    st.markdown("---")
    
    st.markdown("<p style='font-size:16px; font-weight:bold; color:#1C2B4C;'>● Heatmap Correlazioni</p>", unsafe_allow_html=True)
    corr_matrix = df_elaborato.select_dtypes(include=[np.number]).corr()
    
    fig_heat = px.imshow(
        corr_matrix, 
        text_auto='.2f', 
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    fig_heat.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        coloraxis_colorbar_len=0.7,
        title_text='Passa sopra col mouse (Interattiva)', title_x=0.5
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<p style='font-size:16px; font-weight:bold; color:#1C2B4C; margin-top:20px;'>● Distribuzione Voto Target</p>", unsafe_allow_html=True)
    fig_dist = px.histogram(df_originale, x=target_col, color_discrete_sequence=['#2979FF'])
    fig_dist.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Punteggio d'Esame", yaxis_title="Conteggio"
    )
    st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})

# ------------------------------------------
# SCHEDA 2: PREDICTOR DASHBOARD / SIMULATORE
# ------------------------------------------
with tab2:
    st.markdown("<h2 class='data-profiling-title'>Simulatore Predittivo Real-Time</h2>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<p style='font-size: 16px; font-weight: bold; color: #1C2B4C;'>📝 Input Form: Profilo Studente</p>", unsafe_allow_html=True)
        
        in_eta = st.number_input("Età Anagrafica", min_value=15, max_value=90, value=20, format="%d")
        in_genere = st.selectbox("Genere", opzioni_menu.get('gender', ['male', 'female']))
        in_corso = st.selectbox("Corso Frequentato", opzioni_menu.get('course', ['b.tech', 'b.sc', 'b.com']))
        
        in_ore_studio = st.slider("Ore di Studio Giorni", 0.0, 12.0, 4.0, step=0.5)
        in_presenza = st.slider("Frequenza Lezioni %", 0.0, 100.0, 80.0, step=1.0)
        in_qualita_sonno = st.selectbox("Qualità del Sonno", opzioni_menu.get('sleep_quality', ['good', 'average', 'poor']))
        in_ore_sonno = st.slider("Ore Sonno Notturne", 4.0, 12.0, 7.0, step=0.5)
        in_metodo = st.selectbox("Metodo di Studio", opzioni_menu.get('study_method', ['self-study', 'group study']))
        
    st.markdown("<br>", unsafe_allow_html=True)
    submit_sim = st.button("🚀 CALCOLA PREVISIONE", use_container_width=True)

    if submit_sim:
        dati_simulati = {
            'age': in_eta, 'gender': in_genere, 'course': in_corso, 'study_hours': in_ore_studio,
            'class_attendance': in_presenza, 'sleep_hours': in_ore_sonno, 'sleep_quality': in_qualita_sonno,
            'study_method': in_metodo, 'Carico_Totale_Ore': in_ore_studio + in_ore_sonno
        }
        input_user_df = pd.DataFrame([dati_simulati])
        
        for col in input_user_df.columns:
            if col in codici_categorie:
                valore = str(input_user_df[col].iloc[0]).strip()
                lista_cat = list(codici_categorie[col])
                input_user_df[col] = lista_cat.index(valore) if valore in lista_cat else 0
                
        input_user_df = input_user_df.reindex(columns=X.columns, fill_value=0)
        voto_predetto = modello_rf.predict(input_user_df)[0]
        
        with st.container():
            st.markdown("""
                <div style="background-color: #FFFFFF; padding: 20px; border-radius: 20px; margin-top:10px; border: 1px solid rgba(0,0,0,0.03);">
                    <p style="color: rgba(28,43,76,0.7); font-size:14px; margin:0;">🎯 Verdetto Predittivo</p>
                    <p style="color: #1C2B4C; font-size: 32px; font-weight: bold; margin: 5px 0 10px 0;">{pred:.2f} / 100</p>
                </div>
            """.format(pred=voto_predetto), unsafe_allow_html=True)
            
            if voto_predetto >= 70:
                st.success("Rendimento Elevato: Ottima proiezione accademica.")
            elif voto_predetto >= 50:
                st.warning("Rendimento Medio: Margini di miglioramento stabili.")
            else:
                st.error("Rendimento Critico: Si consiglia di rivedere la pianificazione.")
                
        with st.container():
            st.markdown("<p style='font-size:16px; font-weight:bold; color:#1C2B4C; margin-top:20px;'>● Feature Importance (Top 3)</p>", unsafe_allow_html=True)
            importanze = modello_rf.feature_importances_
            df_features = pd.DataFrame({'Fattore': X.columns, 'Importanza': importanze})
            top_3 = df_features.sort_values(by='Importanza', ascending=True).head(3)
            
            fig_bar = px.bar(top_3, x='Importanza', y='Fattore', orientation='h', color='Importanza', color_continuous_scale='Viridis')
            fig_bar.update_layout(margin=dict(l=20, r=20, t=10, b=20), xaxis_title="Importanza", yaxis_title="")
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

# ------------------------------------------
# SCHEDA 3: PERFORMANCE MODELLI
# ------------------------------------------
with tab3:
    st.markdown("<h2 class='data-profiling-title'>Performance dei Modelli</h2>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<p style='font-size:18px; font-weight:bold; color:#1C2B4C;'>Approccio 1: Random Forest</p>", unsafe_allow_html=True)
        st.metric("Errore RMSE", f"{rmse_rf:.4f}")
        
        st.markdown("<p style='font-size:18px; font-weight:bold; color:#1C2B4C; margin-top:20px;'>Approccio 2: Linear Regression</p>", unsafe_allow_html=True)
        st.metric("Errore RMSE", f"{rmse_lr:.4f}")
            
    st.markdown("---")
    st.info(f"Il modello Random Forest registra un errore inferiore di **{abs(rmse_lr - rmse_rf):.4f}** punti.")

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("Ispeziona un'anteprima dei dati storici"):
    st.dataframe(df_originale.head(10), use_container_width=True)
