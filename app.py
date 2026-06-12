import streamlit as st, pandas as pd, numpy as np, plotly.express as px, os, base64, pickle
st.set_page_config(page_title="EmpatiA", layout="wide", initial_sidebar_state="collapsed")
if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("🔒 Accesso Riservato")
        with st.form("L"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("ACCEDI", use_container_width=True):
                if u == "admin" and p == "ia2026": st.session_state["auth"] = True; st.rerun()
                else: st.error("Errato")
    st.stop()
st.markdown("""<style>#MainMenu,footer,header,div[data-testid="stHeader"]{visibility:hidden;display:none!important;}.stApp{background-color:#F8FAFC;color:#1C2B4C;}div.stButton>button{background-color:#1E3A8A!important;color:white!important;font-weight:bold!important;}</style>""", unsafe_allow_html=True)
try:
    with open("logo.png", "rb") as im: b64 = base64.b64encode(im.read()).decode()
    st.markdown(f'<div style="text-align:center;background:linear-gradient(135deg,#1E3A8A,#0F172A);padding:15px;border-radius:12px;"><img src="data:image/png;base64,{b64}" style="max-width:110px;background:white;padding:5px;border-radius:5px;"><h4 style="color:white;margin:5px 0 0 0;">Student Predictor AI</h4></div>', unsafe_allow_html=True)
except: st.title("🎓 Student Predictor AI")
p = "train_leggero.csv" if os.path.exists("train_leggero.csv") else "train.csv"
df = pd.read_csv(p).drop(columns=['id'], errors='ignore').drop_duplicates().fillna(0)
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
    m_rf = RandomForestRegressor(n_estimators=20, max_depth=6, random_state=42, n_jobs=-1).fit(X, df_e.iloc[:,-1])
t1, t2 = st.tabs(["📊 EDA", "🔮 Simulatore"])
with t1:
    st.subheader("Data Profiling")
    st.metric("Righe", df.shape[0]); st.metric("Colonne", df.shape[1])
    g1, g2 = st.columns(2)
    with g1: st.plotly_chart(px.imshow(df_e.select_dtypes(include=[np.number]).corr(), text_auto='.2f', color_continuous_scale='Blugrn').update_layout(height=300), use_container_width=True)
    with g2: st.plotly_chart(px.histogram(df, x=df.columns[-1], color_discrete_sequence=['#FF6F61']).update_layout(height=300), use_container_width=True)
with t2:
    st.subheader("Simulatore Real-Time")
    c1, c2, c3 = st.columns(3)
    with c1:
        v_ag = st.number_input("Età", 15, 90,
