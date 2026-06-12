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
# 1. CONFIGURAZIONE DELLA PAGINA
# ==========================================
st.set_page_config(
    page_title="Student Predictor AI",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🎓"
)

# ==========================================
# 2. CSS PULITO - TITOLI PICCOLI ED ELEGANTI
# ==========================================
st.markdown("""
    <style>
        :root { 
            --primary-bg: #FFFFFF; 
            --text-dark: #1C2B4C; 
            --text-light: #5A6D88; 
            --accent: #2979FF; 
        }
        .stApp { 
            background-color: var(--primary-bg); 
            color: var(--text-dark); 
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; 
        }
        
        /* Nasconde i menu di default di Streamlit per un look più professionale */
        #MainMenu, footer, header, div[data-testid="stHeader"] { visibility: hidden; display: none !important; }
        
        /* HEADER PRINCIPALE IN PURO CODICE (Look moderno blu scuro) */
        .main-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
            padding: 25px;
            border-radius: 16px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        .main-header h1 { color: white !important; font-size: 28px !important; margin: 0 !important; font-weight: 700; }
        .main-header p { color: #93C5FD !important; font-size: 14px !important; margin: 5px 0 0 0 !important; }
        
        /* RICHIESTA 3: TITOLI INTERNI PICCOLI E PROPORZIONATI */
        .section-title { 
            font-size: 20px !important; 
            font-weight: 700 !important; 
            color: var(--text-dark); 
            margin: 15px 0 5px 0 !important; 
        } 
        .section-subtitle {
            font-size: 14px !important;
            font-weight: 600;
            color: var(--text-light);
            margin-bottom: 15px !important;
        }
        
        /* Card per metriche e form */
        .stMetric { background-color: #F8FAFC; border-radius: 12px; padding: 15px; border: 1px solid #E2E8F0; } 
        .form-card { background-color: #F8FAFC; border-radius: 12px; padding: 20px; border: 1px solid #E2E8F0; margin-bottom: 15px; }
        
        /* Pulsante calcola grande e arrotondato */
        .stButton > button { 
            background-color: var(--accent) !important; 
            color: white !important; 
            font-size: 15px !important; 
            font-weight: bold !important; 
            border-radius: 12px !important; 
            height: 45px !important; 
            border: none !important; 
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SCHERMATA DI LOGIN
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
        <div style="text-align: center; margin-top: 50px; background-color: #FFFFFF; padding: 30px; border-radius: 16px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <h2 style="margin: 0; color: #1C2B4C; font-size: 24px;">🔒 Accesso Riservato</h2>
            <p style="color: #5A6D88; margin: 10px 0 0 0; font-size:14px;">Inserisci le credenziali per sbloccare l'applicazione.</p>
        </div>
        <br>
    """, unsafe_allow_html=True)
    
    with st.form("Login Form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("SBLOCCA APPLICAZIONE 🚀", use_container_width=True)
        
        if submit:
            if username == "admin" and password == "ia2026":
                st.session_state["authenticated"] = True
                st.success("Accesso eseguito!")
                st.rerun()
            else:
                st.error("Credenziali errate.")
    st.stop()

# ==========================================
# 4. CARICAMENTO E PREPARAZIONE DATI
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    nome_file = "train_leggero.csv" if os.path.exists("train_leggero.csv") else "train.csv"
    df = pd.read_csv(nome_file)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df = df.drop_duplicates()
    
    colonne_testo = df.select_dtypes(include=['object']).columns
    colonne_numeriche = df.select_dtypes(exclude=['object']).columns
    
    for col in colonne_numeriche:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
        
    if 'study_hours' in df.columns and 'sleep_hours' in df.columns:
        df['Carico_Totale_Ore'] = df['study_hours'] + df['sleep_hours']
    
    opzioni = {}
    for col in colonne_testo:
        df[col] = df[col].fillna('Sconosciuto').astype(str).str.strip()
        opzioni[col] = df[col].unique()
        
    df_encoded = df.copy()
    mappature = {}
    for col in colonne_testo:
        categorie = df_encoded[col].astype('category').cat.categories
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
        mappature[col] = categorie
        
    return df, df_encoded, opzioni, mappature

df_orig, df_enc, opzioni_menu, mappature_cat = load_and_preprocess_data()
target_col = df_enc.columns[-2] if 'Carico_Totale_Ore' in df_enc.columns else df_enc.columns[-1]

X = df_enc.drop(columns=[target_col])
y = df_enc[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 5. ADDESTRAMENTO MODELLI
# ==========================================
@st.cache_resource
def train_models(X_t, X_v, y_t, y_v):
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_t, y_t)
    rmse_rf = np.sqrt(mean_squared_error(y_v, rf.predict(X_v)))
    
    lr = LinearRegression()
    lr.fit(X_t, y_t)
    rmse_lr = np.sqrt(mean_squared_error(y_v, lr.predict(X_v)))
    
    return rf, lr, rmse_rf, rmse_lr

modello_rf, modello_lr, rmse_rf, rmse_lr = train_models(X_train, X_test, y_train, y_test)

# ==========================================
# 6. INTERFACCIA GRAFICA & NAVIGAZIONE TAB
# ==========================================
# Header elegante in codice
st.markdown("""
    <div class="main-header">
        <h1>Student Predictor AI</h1>
        <p>Analisi e Modellazione Predittiva Carriere • Utente: admin</p>
    </div>
""", unsafe_allow_html=True)

# Navigazione nativa Streamlit stabile e funzionante
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔮 Simulatore", "📈 Performance"])

# ------------------------------------------
# TAB 1: ANALISI ESPLORATIVA (EDA)
# ------------------------------------------
with tab1:
    st.markdown("<p class='section-title'>Analisi Esplorativa (EDA)</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Data Profiling del Dataset</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Numero Totale Righe", df_orig.shape[0])
    with c2:
        st.metric("Numero di Colonne", df_orig.shape[1])
    
    st.markdown("<p class='section-title'>📊 Mappa delle Correlazioni</p>", unsafe_allow_html=True)
    corr_matrix = df_enc.select_dtypes(include=[np.number]).corr()
    fig_heat = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='Blugrn', aspect="auto")
    fig_heat.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<p class='section-title'>📊 Distribuzione del Voto Finale</p>", unsafe_allow_html=True)
    fig_dist = px.histogram(df_orig, x=df_orig.columns[-1], color_discrete_sequence=['#2979FF'])
    fig_dist.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Punteggio d'Esame", yaxis_title="Frequenza")
    st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})

# ------------------------------------------
# TAB 2: SIMULATORE PREDITTIVO
# ------------------------------------------
with tab2:
    st.markdown("<p class='section-title'>Simulatore Predittivo Real-Time</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">📝 Inserisci il Profilo dello Studente</p>', unsafe_allow_html=True)
    
    in_eta = st.number_input("Età Anagrafica", min_value=15, max_value=90, value=20)
    in_genere = st.selectbox("Genere", opzioni_menu.get('gender', ['male', 'female']))
    in_corso = st.selectbox("Corso Frequentato", opzioni_menu.get('course', ['b.tech', 'b.sc', 'b.com']))
    in_ore_studio = st.slider("Ore di Studio Giornaliere", 0.0, 12.0, 4.0, step=0.5)
    in_presenza = st.slider("Frequenza Lezioni %", 0.0, 100.0, 80.0, step=1.0)
    in_qualita_sonno = st.selectbox("Qualità del Sonno", opzioni_menu.get('sleep_quality', ['good', 'average', 'poor']))
    in_ore_sonno = st.slider("Ore Sonno Notturne", 4.0, 12.0, 7.0, step=0.5)
    in_metodo = st.selectbox("Metodo di Studio", opzioni_menu.get('study_method', ['self-study', 'group study']))
    st.markdown('</div>', unsafe_allow_html=True)
        
    submit_sim = st.form_submit_button if False else st.button("🚀 CALCOLA PREVISIONE", use_container_width=True)

    if submit_sim:
        dati_simulati = {
            'age': in_eta, 'gender': in_genere, 'course': in_corso, 'study_hours': in_ore_studio,
            'class_attendance': in_presenza, 'sleep_hours': in_ore_sonno, 'sleep_quality': in_qualita_sonno,
            'study_method': in_metodo
        }
        if 'Carico_Totale_Ore' in X.columns:
            dati_simulati['Carico_Totale_Ore'] = in_ore_studio + in_ore_sonno
            
        input_df = pd.DataFrame([dati_simulati])
        
        for col in input_df.columns:
            if col in mappature_cat:
                valore = str(input_df[col].iloc[0]).strip()
                lista_cat = list(mappature_cat[col])
                input_df[col] = lista_cat.index(valore) if valore in lista_cat else 0
                
        input_df = input_df.reindex(columns=X.columns, fill_value=0)
        voto_predetto = modello_rf.predict(input_df)[0]
        
        st.markdown(f"""
            <div style="background-color: #F1F5F9; padding: 20px; border-radius: 12px; margin-top:15px; border: 1px solid #CBD5E1; text-align: center;">
                <p style="color: #5A6D88; font-size:14px; margin:0;">🎯 Punteggio Predetto</p>
                <p style="color: #1E3A8A; font-size: 36px; font-weight: bold; margin: 5px 0 0 0;">{voto_predetto:.2f} / 100</p>
            </div>
        """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 3: PERFORMANCE DEI MODELLI
# ------------------------------------------
with tab3:
    st.markdown("<p class='section-title'>Performance dei Modelli Predittivi</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p class='section-subtitle'>Modello Avanzato (Random Forest)</p>", unsafe_allow_html=True)
        st.metric("Errore Medio RMSE", f"{rmse_rf:.4f}")
    with col2:
        st.markdown("<p class='section-subtitle'>Modello Base (Linear Regression)</p>", unsafe_allow_html=True)
        st.metric("Errore Medio RMSE", f"{rmse_lr:.4f}")
            
    st.markdown("---")
    st.info(f"Il modello Random Forest riduce l'errore di predizione di **{abs(rmse_lr - rmse_rf):.4f}** punti rispetto alla regressione lineare.")

# Anteprima dati storici sempre accessibile in fondo
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("Ispeziona un'anteprima dei dati storici"):
    st.dataframe(df_orig.head(10), use_container_width=True)
