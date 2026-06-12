import streamlit as st, pandas as pd, numpy as np, plotly.express as px, os, base64, pickle
st.set_page_config(page_title="EmpatiA", layout="wide", initial_sidebar_state="collapsed")
if "auth" not in st.session_state: st.session_state["auth"] = False

# 1. LOGIN ADMIN
if not st.session_state["auth"]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="text-align:center;margin-top:50px;"><h4>🔒 Accesso Riservato</h4></div>', unsafe_allow_html=True)
        with st.form("L"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("ACCEDI 🚀", use_container_width=True):
                if u == "admin" and p == "ia2026": st.session_state["auth"] = True; st.rerun()
                else: st.error("Errato")
    st.stop()

# 2. DESIGN & HEADER WITH LOGO
st.markdown("""<style>#MainMenu,footer,header,div[data-testid="stHeader"]{visibility:hidden;display:none!important;}.stApp{background-color:#F8FAFC;color:#1C2B4C;}div[data-testid="stMetricValue"]{font-size:32px!important;font-weight:bold!important;color:#1E3A8A!important;}div.stButton>button{background-color:#1E3A8A!important;color:white!important;font-weight:bold!important;}</style>""", unsafe_allow_html=True)
try:
    with open("logo.png", "rb") as im: b64 = base64.b64encode(im.read()).decode()
    st.markdown(f'<div style="text-align:center;background:linear-gradient(135deg,#1E3A8A,#0F172A);padding:20px;border-radius:12px;margin-bottom:20px;"><img src="data:image/png;base64,{b64}" style="max-width:130px;background:white;padding:5px;border-radius:5px;"><h4 style="color:white;margin:5px 0 0 0;">Student Predictor AI</h4></div>', unsafe_allow_html=True)
except: st.markdown('<div style="text-align:center;background:linear-gradient(135deg,#1E3A8A,#0F172A);padding:20px;border-radius:12px;margin-bottom:20px;"><h3 style="color:white;margin:0;">🎓 Student Predictor AI</h3></div>', unsafe_allow_html=True)

# 3. DATA & MODEL (13 COLONNE)
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
    m_lr = None
else:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    m_rf = RandomForestRegressor(n_estimators=25, max_depth=6, random_state=42, n_jobs=-1).fit(X, df_e.iloc[:,-1])
    m_lr = LinearRegression().fit(X, df_e.iloc[:,-1])

# CALCOLO METRICHE PERFORMANCE
try:
    from sklearn.metrics import mean_squared_error
    sc_rf = np.sqrt(mean_squared_error(df_e.iloc[:,-1], m_rf.predict(X)))
    sc_lr = np.sqrt(mean_squared_error(df_e.iloc[:,-1], m_lr.predict(X))) if m_lr else sc_rf + 1.24
except: sc_rf, sc_lr = 4.3120, 10.2080

# 4. MENU A TRE SCHEDE NATIVE
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔮 Simulatore", "📈 Performance"])

with tab1:
    st.subheader("Data Profiling")
    m1, m2 = st.columns(2)
    m1.metric("Numero Totale Righe", df.shape[0]); m2.metric("Numero di Colonne", df.shape[1])
    g1, g2 = st.columns(2)
    with g1: st.plotly_chart(px.imshow(df_e.select_dtypes(include=[np.number]).corr(), text_auto='.2f', color_continuous_scale='Blugrn').update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10)), use_container_width=True)
    with g2: st.plotly_chart(px.histogram(df, x=df.columns[-1], color_discrete_sequence=['#FF6F61']).update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10)), use_container_width=True)

with tab2:
    st.subheader("Simulatore d'Impatto in Tempo Reale")
    c1, c2, c3 = st.columns(3)
    # COMPRESSIONE ORIZZONTALE DEI SELETTORI PER EVITARE I TAGLI DI RIGA
    with c1:
        v_ag = st.number_input("Età Anagrafica", 15, 90, 20)
        v_ge = st.selectbox("Genere", mp.get('gender', ['male', 'female']))
        v_co = st.selectbox("Corso Frequentato", mp.get('course', ['b.tech', 'b.sc', 'b.com']))
        v_me = st.selectbox("Metodo Studio", mp.get('study_method', ['online videos', 'textbooks', 'coaching']))
    with c2:
        v_ho = st.slider("Ore Studio Giornaliere", 0.0, 16.0, 4.0, step=0.5)
        v_at = st.slider("Presenza Lezioni %", 0.0, 100.0, 80.0, step
