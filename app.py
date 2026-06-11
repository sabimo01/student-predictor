import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Configurazione Pagina con tema scuro/futuristico supportato di serie
st.set_page_config(
    page_title="Student Predictor AI", 
    page_icon="🎓",
    layout="wide"
)

# Header Principale con Stile
st.markdown("""
    <div style="background-gradient: linear-gradient(135deg, #1572B6, #00875A); padding: 20px; border-radius: 10px; margin-bottom: 25px;">
        <h1 style="color: white; margin: 0; text-align: center;">🎓 Student Predictor AI</h1>
        <p style="color: #E0E0E0; margin: 5px 0 0 0; text-align: center; font-size: 1.1rem;">
            Dashboard Predittiva Avanzata basata su Machine Learning per la Stima delle Performance Scolastiche
        </p>
    </div>
""", unsafe_allow_html=True)

# Carichiamo il file di allenamento in modo super veloce
@st.cache_data
def carica_e_allena():
    train_df = pd.read_csv("train_leggero.csv")
    
    if 'id' in train_df.columns:
        train_df = train_df.drop(columns=['id'])
    
    colonna_target = train_df.columns[-1]
    
    colonne_testo = train_df.select_dtypes(include=['object']).columns
    colonne_numeriche = train_df.select_dtypes(exclude=['object']).columns
    
    df_numerico = train_df.copy()
    
    for col in colonne_numeriche:
        df_numerico[col] = pd.to_numeric(df_numerico[col], errors='coerce')
        df_numerico[col] = df_numerico[col].fillna(df_numerico[col].mean())
        
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
    with st.spinner("L'Intelligenza Artificiale sta calibrando i modelli... 🧠"):
        modello, mappature_input, train_df, colonna_target = carica_e_allena()

    # Organizzazione in due macro-sezioni con i box (st.container)
    with st.container(border=True):
        st.markdown("### 🔮 Pannello Configurazione Studente")
        st.write("Modifica i parametri sottostanti per simulare il profilo di uno studente.")
        
        # Divisione pulita in due macro colonne principali
        sinistra, destra = st.columns(2, gap="large")
        
        def ottieni_opzioni(nome_colonna, default_lista):
            if nome_colonna in train_df.columns:
                return train_df[nome_colonna].dropna().unique()
            return default_lista

        with sinistra:
            st.markdown("**Anagrafica e Percorso**")
            eta = st.number_input("Età dello studente", min_value=15, max_value=100, value=20)
            genere = st.selectbox("Genere", ottieni_opzioni('gender', ['male', 'female']))
            corso = st.selectbox("Corso di Studi", ottieni_opzioni('course', ['b.tech', 'b.sc', 'b.com']))
            metodo_studio = st.selectbox("Metodo di studio prevalente", ottieni_opzioni('study_method', ['self-study', 'group study']))

        with destra:
            st.markdown("**Abitudini e Frequenza**")
            ore_studio = st.slider("Ore di studio giornaliere", 0.0, 12.0, 4.0, step=0.5)
            presenza = st.slider("Frequenza delle lezioni (%)", 0.0, 100.0, 75.0, step=1.0)
            ore_sonno = st.slider("Ore di sonno notturne", 4.0, 12.0, 7.0, step=0.5)
            qualita_sonno = st.selectbox("Qualità del sonno percepita", ottieni_opzioni('sleep_quality', ['good', 'average', 'poor']))

    st.markdown("<br>", unsafe_allow_html=True)

    # Area del Pulsante centrata e stilizzata
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        attiva_previsione = st.button("🚀 ELABORA PREVISIONE AI", use_container_width=True)

    # Sezione Verdetto Finale
    if attiva_previsione:
        dati_utente = {
            'age': eta, 'gender': genere, 'course': corso, 'study_hours': ore_studio,
            'class_attendance': presenza, 'sleep_hours': ore_sonno, 'sleep_quality': qualita_sonno,
            'study_method': metodo_studio
        }
        
        input_df = pd.DataFrame([dati_utente])
        
        for col in input_df.columns:
            if col in mappature_input:
                valore_cercato = str(input_df[col].iloc[0]).strip()
                lista_cat = list(mappature_input[col])
                if valore_cercato in lista_cat:
                    input_df[col] = lista_cat.index(valore_cercato)
                else:
                    input_df[col] = 0
                    
        colonne_allenamento = train_df.drop(columns=[colonna_target], errors='ignore').columns
        input_df = input_df.reindex(columns=colonne_allenamento, fill_value=0)
        
        previsione_numerica = modello.predict(input_df)[0]
        
        # Mostriamo il risultato dentro un bellissimo box evidenziato
        st.markdown("---")
        with st.container(border=True):
            st.markdown(f"### 🎯 Risultato Analisi Predittiva")
            
            col_metric, col_text = st.columns([1, 2])
            with col_metric:
                # La metrica gigante stile cruscotto
                st.metric(label="Punteggio Esame Stimato (0-100)", value=f"{previsione_numerica:.1f}/100")
            
            with col_text:
                st.write("**Nota del Modello:**")
                if previsione_numerica >= 75:
                    st.success("L'algoritmo rileva un profilo ad altissimo rendimento. Le abitudini attuali supportano pienamente il successo accademico.")
                elif previsione_numerica >= 55:
                    st.warning("Profilo in linea con la media. Ottimizzando le ore di sonno o la frequenza delle lezioni è possibile incrementare il punteggio stimato.")
                else:
                    st.error("Attenzione: Il modello evidenzia potenziali elementi di rischio (es. sonno insufficiente o bassa frequenza). Si consiglia di rivedere la pianificazione.")

    # Sezione Espandibile per Esplorare i Dati originali in fondo
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.expander("📊 Clicca qui per ispezionare un'anteprima dei dati storici d'allenamento"):
        st.dataframe(train_df.head(10), use_container_width=True)

except Exception as e:
    st.error(f"Si è verificato un errore durante l'allenamento dell'AI: {e}")