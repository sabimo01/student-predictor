import streamlit as st, pandas as pd, numpy as np, plotly.express as px, os, pickle
st.set_page_config(page_title="EmpatiA", layout="wide")
if "auth" not in st.session_state: st.session_state["auth"] = False

if not st.session_state["auth"]:
    with st.form("L"):
        u, p = st.text_input("User"), st.text_input("Pass", type="password")
        if st.form_submit_button("ACCEDI") and u == "admin" and p == "ia2026":
            st.session_state["auth"] = True; st.rerun()
    st.stop()

fl = "train_leggero.csv" if os.path.exists("train_leggero.csv") else "train.csv"
df = pd.read_csv(fl).drop(columns=['id'], errors='ignore').drop_duplicates().fillna(0)
df.columns = [c.strip().lower() for c in df.columns]
df_e = df.copy()
mp = {}
for c in df_e.select_dtypes(include=['object']).columns:
    mp[c] = list(df_e[c].astype('category').cat.categories)
    df_e[c] = df_e[c].astype('category').cat.codes
X = df_e.drop(columns=[df_e.columns[-1]])

if os.path.exists("modello_pronto.pkl"):
    with open("modello_pronto.pkl", "rb") as f: m_rf = pickle.load(f)
else:
    from sklearn.ensemble import RandomForestRegressor
    m_rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1).fit(X, df_e.iloc[:,-1])

t1, t2 = st.tabs(["📊 EDA", "🔮 Simulatore"])
with t1:
    st.metric("Dati storici caricati (Righe)", df.shape[0])
    st.plotly_chart(px.imshow(df_e.select_dtypes(include=[np.number]).corr(), text_auto='.2f', color_continuous_scale='Blugrn').update_layout(height=280), use_container_width=True)
    st.plotly_chart(px.histogram(df, x=df.columns[-1], color_discrete_sequence=['#FF6F61']).update_layout(height=280), use_container_width=True)

with t2:
    st.subheader("Simulatore Automatico")
    m_in = {}
    # IL TRUCCO GENERATIVO: Crea tutti gli input del dataset da solo in automatico!
    for col in X.columns:
        if col in mp: m_in[col] = st.selectbox(f"Seleziona {col.upper()}", mp[col])
        elif df[col].nunique() <= 5: m_in[col] = st.slider(f"Valuta {col.upper()}", int(df[col].min()), int(df[col].max()), int(df[col].median()))
        else: m_in[col] = st.number_input(f"Inserisci {col.upper()}", int(df[col].min()), int(df[col].max()), int(df[col].median()))
        
    if st.button("CALCOLA PREVISIONE VOTO", use_container_width=True):
        in_df = pd.DataFrame([m_in]).reindex(columns=X.columns, fill_value=0)
        for col in in_df.columns:
            if col in mp:
                ls = list(mp[col]); v_s = str(in_df[col].iloc[0]).strip()
                in_df[col] = ls.index(v_s) if v_s in ls else 0
        voto = m_rf.predict(in_df)[0]
        st.success(f"🎯 Voto stimato dall'Intelligenza Artificiale: {voto:.2f} / 100")
