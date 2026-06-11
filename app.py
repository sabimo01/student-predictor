import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Configurazione Pagina
st.set_page_config(page_title="Student Predictor AI", layout="wide")
st.title("🤖 Student Predictor - Intelligenza Artificiale")

# Carichiamo il file di allenamento in modo super veloce
@st.cache_data
def carica_e_allena():
    train_df = pd.read_csv("train.csv")
    
    # Rimuoviamo l'ID se presente
    if 'id' in train_df.columns:
        train_df = train_df.drop(columns=['id'])
    
    colonna_target = train_df.columns[-1]
    
    # Dividiamo le colonne in base al tipo in un colpo solo
    colonne_testo = train_df.select_dtypes(include=['object']).columns
    colonne_numeriche = train_df.select_dtypes(exclude=['object']).columns
    
    # Copia pulita per i calcoli dell'AI
    df_numerico = train_df.copy()
    
    # Gestione ultra-rapida dei numeri
    for col in colonne_numeriche:
        df_numerico[col] = pd.to_numeric(df_numerico[col], errors='coerce')
        df_numerico[col] = df_numerico[col].fillna(df_numerico[col].mean())
        
    # Gestione ultra-rapida dei testi
    mappature_input = {}
    for col in colonne_testo:
        df_numerico[col] = df_numerico[col].fillna('Sconosciuto').astype(str).str.strip()
        categorie = df_numerico[col].astype('category').cat.categories
        df_numerico[col] = df_numerico[col].astype('category').cat.codes
        mappature_input[col] = categorie
            
    X = df_numerico.drop(columns=[colonna_target])
    y = df_numerico[colonna_target]
    
    modello = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    modello.fit(X, y)
    
    return modello, mappature_input, train_df, colonna_target

# Avviamo l'allenamento velocizzato
try:
    with st.spinner("L'Intelligenza Artificiale sta studiando il file train.csv alla massima velocità... ⚡"):
        modello, mappature_input, train_df, colonna_target = carica_e_allena()
    st.success(f"🎉 AI Allenata con successo! Obiettivo: prevedere '{colonna_target}'")

    st.markdown("---")

    # --- INTERFACCIA DI PREVISIONE ---
    st.subheader("🔮 Inserisci i dati di un nuovo studente per calcolare il risultato stimato")

    col1, col2, col3, col4 = st.columns(4)

    # Funzione di supporto per estrarre le opzioni o usare dei default sicuri
    def ottieni_opzioni(nome_colonna, default_lista):
        if nome_colonna in train_df.columns:
            return train_df[nome_colonna].dropna().unique()
        return default_lista

    with col1:
        eta = st.number_input("Età", min_value=15, max_value=100, value=20)
        genere = st.selectbox("Genere", ottieni_opzioni('gender', ['male', 'female']))

    with col2:
        corso = st.selectbox("Corso di Studi", ottieni_opzioni('course', ['b.tech', 'b.sc', 'b.com']))
        ore_studio = st.slider("Ore di studio giornaliere", 0.0, 10.0, 4.0)

    with col3:
        presenza = st.slider("Frequenza lezioni (%)", 0.0, 100.0, 75.0)
        qualita_sonno = st.selectbox("Qualità del sonno", ottieni_opzioni('sleep_quality', ['good', 'average', 'poor']))

    with col4:
        ore_sonno = st.slider("Ore di sonno", 4.0, 12.0, 7.0)
        metodo_studio = st.selectbox("Metodo di studio", ottieni_opzioni('study_method', ['self-study', 'group study']))

    # Bottone per attivare la previsione
    if st.button("🚀 GENERA PREVISIONE AI"):
        dati_utente = {
            'age': eta, 'gender': genere, 'course': corso, 'study_hours': ore_studio,
            'class_attendance': presenza, 'sleep_hours': ore_sonno, 'sleep_quality': qualita_sonno,
            'study_method': metodo_studio
        }
        
        input_df = pd.DataFrame([dati_utente])
        
        # Convertiamo i testi scelti in numeri usando la mappatura salvata
        for col in input_df.columns:
            if col in mappature_input:
                valore_cercato = str(input_df[col].iloc[0]).strip()
                lista_cat = list(mappature_input[col])
                if valore_cercato in lista_cat:
                    input_df[col] = lista_cat.index(valore_cercato)
                else:
                    input_df[col] = 0
                    
        # Sincronizziamo l'ordine delle colonne con l'addestramento
        colonne_allenamento = train_df.drop(columns=[colonna_target], errors='ignore').columns
        input_df = input_df.reindex(columns=colonne_allenamento, fill_value=0)
        
        # Previsione!
        previsione_numerica = modello.predict(input_df)[0]
        
        st.markdown("---")
        st.header(f"Verdetto dell'AI per la colonna **{colonna_target}**:")
        st.info(f"Valore Stimato Previsto dall'AI: **{previsione_numerica:.2f}**")

except Exception as e:
    st.error(f"Si è verificato un errore durante l'allenamento dell'AI: {e}")