import streamlit as st, pandas as pd, numpy as np, plotly.express as px, os, pickle
st.set_page_config(page_title="EmpatiA", layout="wide")
if "auth" not in st.session_state: st.session_state["auth"] = False

if not st.session_state["auth"]:
    with st.form("L"):
        u = st.text_input("User")
        p = st.text_input("Pass", type="password")
        if st.form_submit_button("ACCEDI"):
            if u == "admin" and p == "ia2026": st.session_state["auth"] = True; st.rerun()
            else: st.error("Errato")
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

tab1, tab2 = st.tabs(["📊 EDA", "🔮 Simulatore"])

with tab1:
    st.metric("Righe", df.shape[0]); st.metric("Colonne", df.shape[1])
    g1, g2 = st.columns(2)
    with g1: st.plotly_chart(px.imshow(df_e.select_dtypes(include=[np.number]).corr(), text_auto='.2f', color_continuous_scale='Blugrn').update_layout(height=280), use_container_width=True)
    with g2: st.plotly_chart(px.histogram(df, x=df.columns[-1], color_discrete_sequence=['#FF6F61']).update_layout(height=280), use_container_width=True)

with tab2:
    st.subheader("Simulatore")
    c1, c2, c3 = st.columns(3)
    with c1:
        v_ag = st.number_input("Eta", 15, 90, 20)
        v_ge = st.selectbox("Genere", mp.get('gender', ['male', 'female']))
        v_co = st.selectbox("Corso", mp.get('course', ['b.tech', 'b.sc', 'b.com']))
    with c2:
        v_ho = st.slider("Ore Studio", 0.0, 16.0, 4.0)
        v_at = st.slider("Presenza %", 0.0, 100.0, 80.0)
        v_me = st.selectbox("Metodo", mp.get('study_method', ['online videos', 'textbooks']))
    with c3:
        v_sl = st.slider("Ore Sonno", 3.0, 14.0, 7.0)
        v_sq = st.selectbox("Qualita Sonno", mp.get('sleep_quality', ['good', 'average']))
        v_fa = st.slider("Struttura", 1, 5, 3)

    if st.button("CALCOLA PREVISIONE", use_container_width=True):
        m_in = {'age':v_ag, 'gender':v_ge, 'course':v_co, 'study_hours':v_ho, 'class_attendance':v_at, 'sleep_hours':v_sl, 'sleep_quality':v_sq, 'study_method':v_me, 'facility_rating':v_fa}
        row = {col: m_in.get(col.lower().strip(), 0) for col in X.columns}
        in_df = pd.DataFrame([row]).reindex(columns=X.columns, fill_value=0)
        for col in in_df.columns:
            if col in mp:
                ls = list(mp[col]); v_s = str(in_df[col].iloc[0]).strip()
                in_df[col] = ls.index(v_s) if v_s in ls else 0
        voto = m_rf.predict(in_df)[0]
        st.success(f"🎯 Voto Stimato: {voto:.2f} / 100")

st.write("---")
with st.expander("🔍 Dati Storici"): st.dataframe(df.head(5), use_container_width=True)
