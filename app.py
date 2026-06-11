import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # <-- La nuova super libreria grafica!
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# ==========================================
# 1. AUTENTICAZIONE E AUTORIZZAZIONE
# ==========================================
def login():
    st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
            <h2>🔒 Accesso Piattaforma Student Predictor</h2>
            <p>Inserisci le credenziali autorizzate per accedere al sistema e alla Knowledge Base.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("Login Form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Accedi")
            
            if submit:
                if username == "admin" and password == "ia2026":
                    st.session_state["authenticated"] = True
                    st.success("Accesso eseguito con successo!")
                    st.rerun()
                else:
                    st.error("Credenziali non corrette o non autorizzate.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

# ==========================================
# 2. CONFIGURAZIONE PAGINA E INTERFACCIA
# ==========================================
st.set_page_config(page_title="Student Predictor AI - Piattaforma Pro", layout="wide")

col_logo, col_titolo = st.columns([1, 5])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
    else:
        st.markdown("<h1 style='font-size: 4rem; margin:0;'>🎓</h1>", unsafe_allow_html=True)
with col_titolo:
    st.markdown("""
        <h1 style='margin:0; color:#1572B6;'>Student Predictor AI Dashboard</h1>
        <p style='color:#666; margin:0; font-size:1.1rem;'>Sistema di monitoraggio delle carriere e modellazione predittiva</p>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 3. PIPELINE DI PREPROCESSING & FEATURE ENGINEERING
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("train_leggero.csv")
    
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
# 4. CREAZIONE STRUTTURA A SCHEDE (TABS)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis (EDA)", "🔮 Predictor Dashboard", "📈 Performance Modelli"])

# ------------------------------------------
# SCHEDA 1: EXPLORATORY DATA ANALYSIS (EDA) - VERSIONE INTERATTIVA PLOTLY
# ------------------------------------------
with tab1:
    st.header("Analisi Esplorativa dei Dati (EDA)")
    
    st.subheader("• Data Profiling")
    col_prof1, col_prof2, col_prof3 = st.columns(3)
    with col_prof1:
        st.metric("Numero Totale di Righe", df_originale.shape[0])
    with col_prof2:
        st.metric("Numero di Colonne", df_originale.shape[1])
    with col_prof3:
        st.write("**Tipi di Dati Rilevati:**")
        st.dataframe(df_originale.dtypes.astype(str).to_frame(name="Tipo di Dato"))
        
    st.markdown("---")
    
    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        st.subheader("• Heatmap delle Correlazioni Interattiva")
        corr_matrix = df_elaborato.select_dtypes(include=[np.number]).corr()
        
        # Heatmap ultra-moderna con Plotly
        fig_heat = px.imshow(
            corr_matrix, 
            text_auto='.2f', 
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig_heat.update_layout(title_text='Matrice di Correlazione Organica (Passaci sopra il mouse!)', title_x=0.5)
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col_graf2:
        st.subheader("• Distribuzione della Variabile Target")
        
        # Istogramma interattivo con Plotly
        fig_dist = px.histogram(
            df_originale, 
            x=target_col, 
            kde=True, 
            color_discrete_sequence=['#1572B6']
        )
        fig_dist.update_layout(
            title_text=f"Distribuzione dei Punteggi: {target_col}",
            title_x=0.5,
            xaxis_title="Punteggio d'Esame",
            yaxis_title="Conteggio Studenti"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ------------------------------------------
# SCHEDA 2: PREDICTOR DASHBOARD
# ------------------------------------------
with tab2:
    st.header("Simulatore d'Impatto in Tempo Reale")
    
    with st.container(border=True):
        st.subheader("📝 Input Form: Profilo Studente Ipotetico")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            in_eta = st.number_input("Età Anagrafica", min_value=15, max_value=90, value=20)
            in_genere = st.selectbox("Genere", opzioni_menu.get('gender', ['male', 'female']))
        with c2:
            in_corso = st.selectbox("Corso Frequentato", opzioni_menu.get('course', ['b.tech', 'b.sc', 'b.com']))
            in_ore_studio = st.slider("Ore di Studio Giorni", 0.0, 12.0, 4.0, step=0.5)
        with c3:
            in_presenza = st.slider("Frequenza Lezioni %", 0.0, 100.0, 80.0, step=1.0)
            in_qualita_sonno = st.selectbox("Qualità del Sonno", opzioni_menu.get('sleep_quality', ['good', 'average', 'poor']))
        with c4:
            in_ore_sonno = st.slider("Ore Sonno Notturne", 4.0, 12.0, 7.0, step=0.5)
            in_metodo = st.selectbox("Metodo di Studio", opzioni_menu.get('study_method', ['self-study', 'group study']))
            
    if st.button("🚀 CALCOLA PREVISIONE IN TEMPO REALE", use_container_width=True):
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
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_res, col_feat = st.columns([1, 1], gap="large")
        
        with col_res:
            st.markdown("### 🎯 Verdetto Predittivo")
            st.metric(label="Voto Finale Stimato", value=f"{voto_predetto:.2f} / 100")
            
            if voto_predetto >= 70:
                st.success("Rendimento Elevato: Le abitudini impostate generano una proiezione accademica eccellente.")
            elif voto_predetto >= 50:
                st.warning("Rendimento Medio: Margini di miglioramento stabili bilanciando riposo e presenza.")
            else:
                st.error("Rendimento Critico: Il sistema consiglia di incrementare la presenza e le ore di studio strutturato.")
                
        with col_feat:
            st.markdown("### 📊 Feature Importance (I 3 fattori principali)")
            importanze = modello_rf.feature_importances_
            nomi_features = X.columns
            
            df_features = pd.DataFrame({'Fattore': nomi_features, 'Importanza': importanze})
            top_3 = df_features.sort_values(by='Importanza', ascending=True).head(3) # Ascending true per visualizzazione orizzontale corretta
            
            # Grafico a barre orizzontale moderno con Plotly
            fig_bar = px.bar(
                top_3, 
                x='Importanza', 
                y='Fattore', 
                orientation='h',
                color='Importanza',
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(
                title_text="I 3 fattori chiave che influenzano di più il voto finale",
                title_x=0.5,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------
# SCHEDA 3: PERFORMANCE MODELLI
# ------------------------------------------
with tab3:
    st.header("Validazione e Confronto Algoritmi")
    st.write("Misurazione dell'errore quadratico medio dei modelli matematici per decretare la precisione dell'architettura.")
    
    col_mod1, col_mod2 = st.columns(2)
    with col_mod1:
        with st.container(border=True):
            st.markdown("### 🌲 Approccio 1: Random Forest Regressor")
            st.write("Algoritmo non lineare basato su ensemble di alberi decisionali.")
            st.metric(label="Errore Medio (RMSE)", value=f"{rmse_rf:.4f}")
            st.caption("Un valore di RMSE più basso indica predizioni più accurate e vicine al dato reale.")
            
    with col_mod2:
        with st.container(border=True):
            st.markdown("### 📈 Approccio 2: Regressione Lineare")
            st.write("Modello matematico statistico lineare standard.")
            st.metric(label="Errore Medio (RMSE)", value=f"{rmse_lr:.4f}")
            st.caption("Punto di riferimento classico per stimare i trend quantitativi continui.")
            
    st.markdown("---")
    st.subheader("💡 Verdetto del Team di Data Science")
    
    differenza_calcolata = abs(rmse_lr - rmse_rf)
    
    if rmse_rf < rmse_lr:
        st.info(f"L'algoritmo **Random Forest** è stato scelto come motore principale poiché registra un errore RMSE inferior di **{differenza_calcolata:.4f}** punti rispetto alla Regressione Lineare.")
    else:
        st.info("L'algoritmo **Regressione Lineare** risulta più performante o equivalente in questo specifico sotto-insieme di dati.")

# Anteprima Dati Storici di Knowledge Base
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("📊 Ispeziona un'anteprima dei dati storici d'allenamento (Knowledge Base)"):
    st.dataframe(df_originale.head(10), use_container_width=True)
