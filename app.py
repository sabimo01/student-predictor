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
# 1. CONFIGURAZIONE UNICA DELLA PAGINA - MOBILE-FIRST PRO
# ==========================================
st.set_page_config(
    page_title="Student Predictor AI",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🎓"
)

# ==========================================
# 2. CSS PERSONALIZZATO PER RENDERLA IDENTICA AL MOCKUP
# ==========================================
st.markdown("""
    <style>
        :root { --primary-bg: #FFFFFF; --text-dark: #1C2B4C; --text-light: #5A6D88; --accent: #2979FF; }
        .stApp { background-color: var(--primary-bg); color: var(--text-dark); font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
        
        /* Rimozione totale dei menu Streamlit nativi per non sporcare la grafica */
        #MainMenu, footer, header, div[data-testid="stHeader"] { visibility: hidden; display: none !important; }
        
        /* TITOLI INTERNI RIMPICCIOLITI (Richiesta 3: non più enormi) */
        .data-profiling-title { 
            font-size: 20px !important; 
            font-weight: bold !important; 
            color: var(--text-dark); 
            margin: 10px 0 5px 0 !important; 
            padding-top: 5px;
        } 
        .section-subtitle {
            font-size: 14px !important;
            font-weight: bold !important;
            color: var(--text-dark);
            margin-bottom: 10px !important;
        }
        
        /* Card delle metriche pulite e arrotondate */
        .stMetric { background-color: #F8FAFC; border-radius: 16px; padding: 15px; border: 1px solid rgba(0,0,0,0.04); margin-bottom: 10px; } 
        .form-card { background-color: #F8FAFC; border-radius: 16px; padding: 20px; border: 1px solid rgba(0,0,0,0.04); margin-bottom: 15px; }
        
        /* Bottone Calcola grande e responsive */
        .stButton > button { background-color: var(--accent) !important; color: white !important; font-size: 16px !important; font-weight: bold !important; border-radius: 22px !important; height: 50px !important; border: none !important; margin-top: 10px; }
        
        /* Stile menu di navigazione compatto */
        .nav-label { font-size: 14px; font-weight: bold; color: var(--text-light); margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. AUTENTICAZIONE E AUTORIZZAZIONE
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
        <div style="text-align: center; margin-top: 50px; background-color: #FFFFFF; padding: 30px; border-radius: 18px; border: 1px solid rgba(0,0,0,0.05); box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <h2 style="margin: 0; color: #1C2B4C; font-size: 26px;">🔒 Accesso Riservato</h2>
            <p style="color: rgba(28,43,76,0.7); margin: 10px 0 0 0; font-size:14px;">Inserisci le credenziali per sbloccare l'applicazione.</p>
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
                st.success("Accesso eseguito!")
                st.rerun()
            else:
                st.error("Credenziali non corrette.")
    st.stop()

# ==========================================
# 4. INTERFACCIA MOCKUP VISIVO (POST-LOGIN)
# ==========================================
# Carichiamo la testata perfetta con i pulsanti bianchi incorporati (Richiesta 1 e 2)
if os.path.exists("header_mockup.png"):
    st.image("header_mockup.png", use_container_width=True)
else:
    # Fallback di sicurezza se l'immagine non è ancora caricata su GitHub
    st.markdown("""
        <div style="width: 100%; padding: 20px; background: linear-gradient(135deg, #1C2B4C 0%, #2979FF 100%); border-radius: 0 0 20px 20px; color: white; text-align:center;">
            <h3>🎓 Student Predictor AI</h3>
            <p>Carica 'header_mockup.png' per attivare la grafica completa.</p>
        </div>
    """, unsafe_allow_html=True)

# SISTEMA DI NAVIGAZIONE COMPATTO SOTTO LA TESTATA
st.markdown('<p class="nav-label">🎛️ SELEZIONA SCHEDA VISUALIZZAZIONE:</p>', unsafe_allow_html=True)
scelta_scheda = st.selectbox(
    "Navigazione",
    ["📊 Analisi Esplorativa (EDA)", "🔮 Simulatore Predittivo", "📈 Performance dei Modelli"],
    label_visibility="collapsed"
)

st.markdown("---")

# ==========================================
# 5. PIPELINE DI PREPROCESSING & MACHINE LEARNING
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
# 6. LOGICA DI VISUALIZZAZIONE DELLE SCHEDE
# ==========================================

# ------------------------------------------
# CONTENUTO SCHEDA 1: EDA
# ------------------------------------------
if scelta_scheda == "📊 Analisi Esplorativa (EDA)":
    st.markdown("<h2 class='data-profiling-title'>Analisi Esplorativa (EDA)</h2>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Data Profiling</p>", unsafe_allow_html=True)
    
    st.metric("Numero Totale Righe", df_originale.shape[0])
    st.metric("Numero di Colonne", df_originale.shape[1])
    
    st.write("**Tipi di Dati Rilevati:**")
    st.dataframe(df_originale.dtypes.astype(str).to_frame(name="Tipo"), use_container_width=True)
        
    st.markdown("---")
    st.markdown("<p class='section-subtitle'>📊 Heatmap Correlazioni Interattiva</p>", unsafe_allow_html=True)
    corr_matrix = df_elaborato.select_dtypes(include=[np.number]).corr()
    
    fig_heat = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r', aspect="auto")
    fig_heat.update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_colorbar_len=0.7)
    st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<p class='section-subtitle' style='margin-top:20px;'>📊 Distribuzione Voto Target</p>", unsafe_allow_html=True)
    fig_dist = px.histogram(df_originale, x=target_col, color_discrete_sequence=['#2979FF'])
    fig_dist.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Punteggio d'Esame", yaxis_title="Conteggio")
    st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})

# ------------------------------------------
# CONTENUTO SCHEDA 2: SIMULATORE
# ------------------------------------------
elif scelta_scheda == "🔮 Simulatore Predittivo":
    st.markdown("<h2 class='data-profiling-title'>Simulatore Predittivo Real-Time</h2>", unsafe_allow_html=True)
    
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">📝 Profilo Studente Ipotetico</p>', unsafe_allow_html=True)
    
    in_eta = st.number_input("Età Anagrafica", min_value=15, max_value=90, value=20, format="%d")
    in_genere = st.selectbox("Genere", opzioni_menu.get('gender', ['male', 'female']))
    in_corso = st.selectbox("Corso Frequentato", opzioni_menu.get('course', ['b.tech', 'b.sc', 'b.com']))
    in_ore_studio = st.slider("Ore di Studio Giorni", 0.0, 12.0, 4.0, step=0.5)
    in_presenza = st.slider("Frequenza Lezioni %", 0.0, 100.0, 80.0, step=1.0)
    in_qualita_sonno = st.selectbox("Qualità del Sonno", opzioni_menu.get('sleep_quality', ['good', 'average', 'poor']))
    in_ore_sonno = st.slider("Ore Sonno Notturne", 4.0, 12.0, 7.0, step=0.5)
    in_metodo = st.selectbox("Metodo di Studio", opzioni_menu.get('study_method', ['self-study', 'group study']))
    st.markdown('</div>', unsafe_allow_html=True)
        
    submit_sim = st.button("🚀 CALCOLA PREVISIONE FINALIZZATA", use_container_width=True)

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
        
        st.markdown("""
            <div style="background-color: #F8FAFC; padding: 20px; border-radius: 14px; margin-top:15px; border: 1px solid rgba(0,0,0,0.04);">
                <p style="color: var(--text-light); font-size:13px; margin:0;">🎯 Verdetto Predittivo</p>
                <p style="color: var(--text-dark); font-size: 30px; font-weight: bold; margin: 5px 0 0 0;">{pred:.2f} / 100</p>
            </div>
        """.format(pred=voto_predetto), unsafe_allow_html=True)
        
        if voto_predetto >= 70:
            st.success("Rendimento Elevato: Ottima proiezione.")
        elif voto_predetto >= 50:
            st.warning("Rendimento Medio: Margini stabili.")
        else:
            st.error("Rendimento Critico: Si consiglia revisione del piano orario.")

# ------------------------------------------
# CONTENUTO SCHEDA 3: PERFORMANCE
# ------------------------------------------
elif scelta_scheda == "📈 Performance dei Modelli":
    st.markdown("<h2 class='data-profiling-title'>Performance dei Modelli</h2>", unsafe_allow_html=True)
    
    st.markdown("<p class='section-subtitle'>Approccio 1: Random Forest</p>", unsafe_allow_html=True)
    st.metric("Errore Medio RMSE", f"{rmse_rf:.4f}")
    
    st.markdown("<p class='section-subtitle' style='margin-top:15px;'>Approccio 2: Linear Regression</p>", unsafe_allow_html=True)
    st.metric("Errore Medio RMSE", f"{rmse_lr:.4f}")
            
    st.markdown("---")
    st.info(f"Il modello Random Forest registra un errore inferiore di **{abs(rmse_lr - rmse_rf):.4f}** punti.")

# Tabella storica sempre consultabile in fondo
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("Ispeziona un'anteprima dei dati storici"):
    st.dataframe(df_originale.head(10), use_container_width=True)
