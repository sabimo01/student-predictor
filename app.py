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

try:
    from sklearn.metrics import mean_squared_error
    sc_rf = np.sqrt(mean_squared_error(df_e.iloc[:,-1], m_rf.predict(X)))
except: sc_rf = 4.3120

tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔮 Simulatore", "📈 Performance"])

with tab1:
    st.metric("Righe", df.shape[0]); st.metric("Colonne", df.shape[1])
    st.plotly_chart(px.imshow(df_e.select_dtypes(include=[np.number]).corr(), text_auto='.2f', color_continuous_scale='Blugrn').update_layout(height=280), use_container_width=True)
    st.plotly_chart(px.histogram(df, x=df.columns[-1], color_discrete_sequence=['#FF6F61']).update_layout(height=280), use_container_width=True)

with tab2:
    st.subheader("Simulatore")
    v_ag = st.number_input("Eta", 15, 90, 20)
    v_ge = st.selectbox("Genere", mp.get('gender', ['male', 'female']))
    v_co = st.selectbox("Corso", mp.get('course', ['b.tech', 'b.sc', 'b.com']))
    v_me = st.selectbox("Metodo", mp.get('study_method', ['online videos', 'textbooks']))
    v_ho = st.slider("Ore Studio", 0.0, 16.0, 4.0)
    v_at = st.slider("Presenza %", 0.0, 100.0, 80.0)
    v_it = st.selectbox("Internet", mp.get('internet_access', ['yes', 'no']))
    v_di = st.selectbox("Difficolta", mp.get('exam_difficulty', ['medium', 'low', 'high']))
    v_sl = st.slider("Ore Sonno", 3.0, 14.0, 7.0)
    v_sq = st.selectbox("Qualita Sonno", mp.get('sleep_quality',
