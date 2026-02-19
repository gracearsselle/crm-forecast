import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CRM · Deep Learning Forecast", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;600&display=swap');
:root{--gold:#C9A84C;--dark:#0A0C10;--dark2:#111520;--card:#13192A;--border:#1E2840;--text:#E8EAF0;--muted:#7B8099;--green:#2ECC8B;--red:#E05C5C;--blue:#4A90D9;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--dark)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 2rem 2rem 2rem!important;max-width:1400px;}
.hero{background:linear-gradient(135deg,#0A0C10 0%,#0D1525 50%,#0A0C10 100%);border-bottom:1px solid var(--border);padding:2.5rem 3rem;margin:0 -2rem 2rem -2rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-50%;right:-10%;width:600px;height:600px;background:radial-gradient(circle,rgba(201,168,76,0.08) 0%,transparent 70%);pointer-events:none;}
.hero-title{font-family:'Playfair Display',serif;font-size:3rem;font-weight:900;color:var(--gold);margin:0;line-height:1.1;}
.hero-sub{font-size:1rem;color:var(--muted);margin-top:.5rem;letter-spacing:2px;text-transform:uppercase;font-weight:300;}
.hero-badge{display:inline-block;background:rgba(201,168,76,.12);border:1px solid rgba(201,168,76,.3);color:var(--gold);font-size:.72rem;letter-spacing:2px;text-transform:uppercase;padding:.3rem .8rem;border-radius:2px;margin-bottom:1rem;font-family:'JetBrains Mono',monospace;}
.metric-row{display:flex;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap;}
.metric-card{flex:1;min-width:140px;background:var(--card);border:1px solid var(--border);border-radius:6px;padding:1.25rem 1.5rem;position:relative;overflow:hidden;}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--gold),transparent);}
.metric-label{font-size:.72rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);font-family:'JetBrains Mono',monospace;margin-bottom:.5rem;}
.metric-value{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:700;color:var(--text);line-height:1;}
.metric-delta{font-size:.82rem;margin-top:.35rem;font-family:'JetBrains Mono',monospace;}
.delta-pos{color:var(--green);} .delta-neg{color:var(--red);}
.section-header{display:flex;align-items:center;gap:1rem;margin:2rem 0 1rem 0;padding-bottom:.75rem;border-bottom:1px solid var(--border);}
.section-title{font-family:'Playfair Display',serif;font-size:1.4rem;color:var(--text);margin:0;}
.section-pill{background:rgba(201,168,76,.1);border:1px solid rgba(201,168,76,.25);color:var(--gold);font-size:.68rem;letter-spacing:1.5px;text-transform:uppercase;padding:.2rem .6rem;border-radius:2px;font-family:'JetBrains Mono',monospace;}
[data-testid="stSidebar"]{background:var(--dark2)!important;border-right:1px solid var(--border)!important;}
.sidebar-logo{font-family:'Playfair Display',serif;font-size:1.3rem;color:var(--gold);font-weight:700;margin-bottom:.25rem;}
.sidebar-tagline{font-size:.72rem;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid var(--border);}

/* ── Boîte résultat de prédiction ── */
.pred-result{background:linear-gradient(135deg,rgba(201,168,76,.12),rgba(201,168,76,.04));border:2px solid rgba(201,168,76,.5);border-radius:10px;padding:2.5rem;text-align:center;margin:1rem 0;}
.pred-result-label{font-size:.8rem;letter-spacing:3px;text-transform:uppercase;color:var(--muted);font-family:'JetBrains Mono',monospace;margin-bottom:1rem;}
.pred-result-price{font-family:'Playfair Display',serif;font-size:4rem;font-weight:900;color:var(--gold);line-height:1;}
.pred-result-delta-pos{color:#2ECC8B;font-family:'JetBrains Mono',monospace;font-size:1.1rem;margin-top:.5rem;}
.pred-result-delta-neg{color:#E05C5C;font-family:'JetBrains Mono',monospace;font-size:1.1rem;margin-top:.5rem;}
.pred-result-meta{font-size:.75rem;color:var(--muted);margin-top:.75rem;font-family:'JetBrains Mono',monospace;}

/* ── Formulaire input ── */
.form-box{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:1.5rem;margin-bottom:1rem;}
.form-title{font-family:'Playfair Display',serif;font-size:1.1rem;color:var(--gold);margin-bottom:1rem;}
.input-hint{font-size:.72rem;color:var(--muted);font-family:'JetBrains Mono',monospace;margin-top:.25rem;}

/* ── Tableau historique ── */
.stDataFrame{border:1px solid var(--border)!important;border-radius:6px;}

.stTabs [data-baseweb="tab-list"]{background:var(--dark2);border-radius:4px;padding:4px;border:1px solid var(--border);gap:4px;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;border-radius:3px!important;font-size:.8rem!important;letter-spacing:1px;text-transform:uppercase;font-family:'JetBrains Mono',monospace!important;padding:.5rem 1rem!important;border:none!important;}
.stTabs [aria-selected="true"]{background:rgba(201,168,76,.15)!important;color:var(--gold)!important;}
.info-box{background:var(--card);border:1px solid var(--border);border-left:3px solid var(--gold);border-radius:4px;padding:1rem 1.25rem;margin:1rem 0;font-size:.88rem;color:var(--muted);}
.warn-box{background:rgba(224,92,92,.05);border:1px solid rgba(224,92,92,.3);border-left:3px solid #E05C5C;border-radius:4px;padding:1rem 1.25rem;margin:1rem 0;font-size:.88rem;color:var(--muted);}
.stButton>button{background:linear-gradient(135deg,rgba(201,168,76,.2),rgba(201,168,76,.1))!important;border:1px solid rgba(201,168,76,.6)!important;color:var(--gold)!important;border-radius:4px!important;font-family:'JetBrains Mono',monospace!important;font-size:.85rem!important;letter-spacing:1px!important;text-transform:uppercase!important;padding:.6rem 2rem!important;width:100%!important;}
.stButton>button:hover{background:rgba(201,168,76,.3)!important;border-color:var(--gold)!important;}
div[data-testid="stNumberInput"] input{background:#0D1525!important;border:1px solid var(--border)!important;color:var(--text)!important;font-family:'JetBrains Mono',monospace!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  GRU MODEL
# ─────────────────────────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2,
                 output_size=1, dropout=0.2):
        super().__init__()
        self.gru     = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :]))

# ─────────────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────────────
GOLD='#C9A84C'; GREEN='#2ECC8B'; RED='#E05C5C'; BLUE='#4A90D9'; PURPLE='#9B59B6'
SEQ_LEN = 60

def plotly_layout(**kw):
    base = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', color='#7B8099', size=12),
                xaxis=dict(gridcolor='#1E2840', showgrid=True, zeroline=False,
                           tickfont=dict(family='JetBrains Mono', size=10)),
                yaxis=dict(gridcolor='#1E2840', showgrid=True, zeroline=False,
                           tickfont=dict(family='JetBrains Mono', size=10)),
                margin=dict(l=10,r=10,t=40,b=10),
                legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1E2840',
                            borderwidth=1, font=dict(size=11)),
                hovermode='x unified')
    base.update(kw)
    return base

# ─────────────────────────────────────────────────────────────
#  CHARGEMENT MODÈLES
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_gru_model(path='model_gru_salesforce.pth'):
    if not os.path.exists(path): return None, f"Introuvable : {path}"
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd   = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        inp=13; hid=128; nl=2
        for k,v in sd.items():
            if 'gru.weight_ih_l0' in k: hid=v.shape[0]//3; inp=v.shape[1]
        m = GRUModel(input_size=inp, hidden_size=hid, num_layers=nl)
        m.load_state_dict(sd, strict=False); m.eval()
        return m, None
    except Exception as e: return None, str(e)

@st.cache_resource
def load_np_model(path='model_neuralprophet_crm.pkl'):
    if not os.path.exists(path): return None, f"Introuvable : {path}"
    try:
        with open(path,'rb') as f: return pickle.load(f), None
    except:
        try: return torch.load(path, map_location='cpu', weights_only=False), None
        except Exception as e: return None, str(e)

@st.cache_resource
def load_scaler(path='scaler.pkl'):
    if not os.path.exists(path): return None
    try:
        with open(path,'rb') as f: return pickle.load(f)
    except: return None

@st.cache_resource
def load_np_metrics(path='neuralprophet_metrics.pkl'):
    if not os.path.exists(path): return {}
    try:
        with open(path,'rb') as f: m=pickle.load(f)
        km={'MAE':'MAE ($)','RMSE':'RMSE ($)','MAPE':'MAPE (%)','R2':'R²','R²':'R²','MSE':'MSE'}
        return {km.get(k,k):v for k,v in m.items()} if isinstance(m,dict) else {}
    except: return {}

# ─────────────────────────────────────────────────────────────
#  DONNÉES
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_market_data(period='5y'):
    df = yf.download('CRM', period=period, auto_adjust=True, progress=False)
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def compute_features(df):
    d = df.copy()
    d['Returns']    = d['Close'].pct_change()
    d['MA7']        = d['Close'].rolling(7).mean()
    d['MA30']       = d['Close'].rolling(30).mean()
    d['MA90']       = d['Close'].rolling(90).mean()
    d['Volatility'] = d['Close'].pct_change().rolling(30).std()
    d['HL_Spread']  = d['High'] - d['Low']
    d['CO_Ratio']   = d['Close'] / d['Open']
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI']        = 100 - (100/(1+gain/(loss+1e-9)))
    d['Close_Lag1'] = d['Close'].shift(1)
    d['Close_Lag7'] = d['Close'].shift(7)
    return d.dropna().reset_index(drop=True)

def get_feature_cols(scaler, df, gru_model=None):
    cands = ['Open','High','Low','Close','Volume','Returns','MA7','MA30','MA90',
             'Volatility','HL_Spread','CO_Ratio','RSI','Close_Lag1','Close_Lag7']
    avail = [c for c in cands if c in df.columns]
    if scaler is not None and hasattr(scaler,'feature_names_in_'):
        cols=[c for c in scaler.feature_names_in_ if c in df.columns]
        if cols: return cols
    if scaler is not None and hasattr(scaler,'n_features_in_'):
        return avail[:scaler.n_features_in_]
    if gru_model is not None:
        for k,v in gru_model.state_dict().items():
            if 'gru.weight_ih_l0' in k: return avail[:v.shape[1]]
    return avail

# ─────────────────────────────────────────────────────────────
#  PRÉDICTION GRU — sur séquence historique réelle
# ─────────────────────────────────────────────────────────────
def gru_predict_from_sequence(model, sequence_df, scaler, feature_cols):
    """
    sequence_df : DataFrame avec les SEQ_LEN dernières lignes de données réelles
    Retourne le prix prédit J+1 en dollars (dénormalisé).
    """
    feat   = sequence_df[feature_cols].values.astype(np.float32)
    ci     = list(feature_cols).index('Close')

    if scaler is not None:
        feat_s = scaler.transform(feat)
    else:
        mins   = feat.min(axis=0, keepdims=True)
        maxs   = feat.max(axis=0, keepdims=True)
        feat_s = (feat - mins) / (maxs - mins + 1e-9)

    model.eval()
    with torch.no_grad():
        x    = torch.tensor(feat_s, dtype=torch.float32).unsqueeze(0)
        pred = model(x).item()

    # Dénormalisation correcte
    if scaler is not None and hasattr(scaler, 'inverse_transform'):
        dummy = np.zeros((1, len(feature_cols)), dtype=np.float32)
        dummy[0, ci] = pred
        return float(scaler.inverse_transform(dummy)[0, ci])
    else:
        c_min = feat[:, ci].min(); c_max = feat[:, ci].max()
        return float(pred * (c_max - c_min) + c_min)


def gru_predict_testset(model, df, scaler, feature_cols, seq_len=60):
    """Prédiction sur le test set complet pour le graphique."""
    feat = df[feature_cols].values.astype(np.float32)
    ci   = list(feature_cols).index('Close')

    if scaler is not None:
        feat_s = scaler.transform(feat)
    else:
        mins=feat.min(axis=0,keepdims=True); maxs=feat.max(axis=0,keepdims=True)
        feat_s=(feat-mins)/(maxs-mins+1e-9)

    n=len(feat_s); n_test=min(300,n//4); start=n-n_test-seq_len
    if start<0: return None,None,None

    preds,trues,dates=[],[],[]
    model.eval()
    with torch.no_grad():
        for i in range(n_test):
            seq=feat_s[start+i:start+i+seq_len]
            x=torch.tensor(seq,dtype=torch.float32).unsqueeze(0)
            preds.append(model(x).item())
            trues.append(feat_s[start+i+seq_len,ci])
            dates.append(df['Date'].iloc[start+i+seq_len])

    preds=np.array(preds,dtype=np.float32)
    trues=np.array(trues,dtype=np.float32)

    if scaler is not None and hasattr(scaler,'inverse_transform'):
        nf=len(feature_cols)
        dp=np.zeros((len(preds),nf),dtype=np.float32); dp[:,ci]=preds
        dt=np.zeros((len(trues),nf),dtype=np.float32); dt[:,ci]=trues
        preds=scaler.inverse_transform(dp)[:,ci]
        trues=scaler.inverse_transform(dt)[:,ci]
    else:
        cm=feat[:,ci].min(); cx=feat[:,ci].max()
        preds=preds*(cx-cm)+cm
        trues=df['Close'].values[start+seq_len:start+seq_len+n_test]

    return dates, trues.astype(float), preds.astype(float)


def compute_metrics(yt, yp):
    mae  = float(np.mean(np.abs(yt-yp)))
    mse  = float(np.mean((yt-yp)**2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((yt-yp)/(np.abs(yt)+1e-9)))*100)
    r2   = float(1-np.sum((yt-yp)**2)/(np.sum((yt-np.mean(yt))**2)+1e-9))
    return {'MAE ($)':mae,'RMSE ($)':rmse,'MAPE (%)':mape,'R²':r2,'MSE':mse}

# ─────────────────────────────────────────────────────────────
#  PRÉDICTION NEURALPROPHET
# ─────────────────────────────────────────────────────────────
def np_predict(model, df, horizon=1):
    try:
        df_np = df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'}).copy()
        df_np['ds'] = pd.to_datetime(df_np['ds'])

        extra=[]
        for attr in ['extra_lagged_regressors','future_regressors','lagged_regressors','regressors']:
            val=getattr(model,attr,None)
            if val: extra+=list(val.keys())

        rm={'MA7':'MA7','MA21':'MA30','MA30':'MA30','MA90':'MA90','RSI':'RSI',
            'Volume':'Volume','Returns':'Returns','Volatility':'Volatility'}
        for r in extra:
            src=rm.get(r,r)
            col=src if src in df.columns else (r if r in df.columns else None)
            if col: df_np[r]=df[col].values

        df_np=df_np.dropna()
        n_hist=min(300,len(df_np)//4)
        future=model.make_future_dataframe(df_np,periods=horizon,n_historic_predictions=n_hist)
        for r in extra:
            if r in df_np.columns and r not in future.columns:
                lv=float(df_np[r].iloc[-1])
                mg=future[['ds']].merge(df_np[['ds',r]],on='ds',how='left')
                future[r]=mg[r].fillna(lv).values
        return model.predict(future), None
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────
#  HELPERS UI
# ─────────────────────────────────────────────────────────────
def metric_card(label, value, delta=None, prefix='', suffix=''):
    dh=''
    if delta is not None:
        cls='delta-pos' if delta>=0 else 'delta-neg'
        s='▲' if delta>=0 else '▼'
        dh=f'<div class="metric-delta {cls}">{s} {abs(delta):.2f}%</div>'
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{prefix}{value}{suffix}</div>{dh}</div>')

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">📈 CRM Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Deep Learning · Salesforce</div>', unsafe_allow_html=True)
    st.markdown("**Période d'analyse**")
    pm={'1 an':'1y','2 ans':'2y','5 ans':'5y','10 ans':'10y','Max':'max'}
    ps=st.selectbox('Période',list(pm.keys()),index=2,label_visibility='collapsed')
    period=pm[ps]
    st.markdown("**Horizon Monte Carlo**")
    horizon=st.slider('Horizon',1,30,7,label_visibility='collapsed')
    st.markdown("**Indicateurs techniques**")
    show_ma  = st.checkbox('Moyennes mobiles', value=True)
    show_vol = st.checkbox('Volume', value=True)
    show_rsi = st.checkbox('RSI (14)', value=True)
    st.markdown("---")
    st.markdown("""<div style='font-size:.72rem;color:#4A5070;line-height:1.6;font-family:JetBrains Mono,monospace;'>
TSAGNING GRACE<br>GRU + Neural Prophet<br>SALESFORCE (CRM)<br>M2 · NLP · 2024-2025</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""<div class="hero">
  <div class="hero-badge">Deep Learning · Stock Forecast</div>
  <h1 class="hero-title">Salesforce · CRM</h1>
  <p class="hero-sub">GRU Neural Network &amp; Neural Prophet · Prédiction interactive</p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  CHARGEMENT
# ─────────────────────────────────────────────────────────────
with st.spinner('Chargement des données marché...'):
    try:
        df_raw=load_market_data(period); df=compute_features(df_raw); data_ok=True
    except Exception as e:
        st.error(f"Erreur données : {e}"); data_ok=False
if not data_ok: st.stop()

gru_model, gru_err = load_gru_model('model_gru_salesforce.pth')
np_model,  np_err  = load_np_model('model_neuralprophet_crm.pkl')
np_metrics_saved   = load_np_metrics('neuralprophet_metrics.pkl')
scaler             = load_scaler('scaler.pkl')
feature_cols       = get_feature_cols(scaler, df, gru_model)

with st.sidebar:
    st.markdown("---"); st.markdown("**Statut modèles**")
    st.success(f"✅ GRU ({len(feature_cols)} features)") if gru_model else st.error(f"❌ GRU: {gru_err}")
    st.success("✅ Neural Prophet") if np_model else st.error(f"❌ NP: {np_err}")
    st.info(f"⚙️ Scaler ({getattr(scaler,'n_features_in_','?')})") if scaler else st.warning("⚠️ Scaler absent")

# ─────────────────────────────────────────────────────────────
#  KPI
# ─────────────────────────────────────────────────────────────
last=df.iloc[-1]; prev=df.iloc[-2]
change=float((last['Close']-prev['Close'])/prev['Close']*100)
vol7  =float(df['Close'].pct_change().tail(7).std()*np.sqrt(252)*100)
rsi_v =float(df['RSI'].iloc[-1])

row='<div class="metric-row">'
row+=metric_card('Prix Actuel',   f"{float(last['Close']):.2f}", change, '$')
row+=metric_card('Volume (J)',     f"{int(last['Volume']):,}")
row+=metric_card('RSI (14j)',      f"{rsi_v:.1f}")
row+=metric_card('Volatilité 7j',  f"{vol7:.1f}", suffix='%')
row+=metric_card('MA 30j',        f"{float(last['MA30']):.2f}", prefix='$')
row+='</div>'
st.markdown(row, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Marché", "🤖  Prédire", "📉  Techniques", "⚖️  Comparaison", "ℹ️  Modèles"
])

# ══════════════════════════════════════════════════════════════
#  TAB 1 · MARCHÉ
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header"><h2 class="section-title">Cours Historique</h2><span class="section-pill">OHLC · Candlestick</span></div>', unsafe_allow_html=True)

    fig_c=go.Figure()
    fig_c.add_trace(go.Candlestick(x=df_raw.tail(500)['Date'],
        open=df_raw.tail(500)['Open'], high=df_raw.tail(500)['High'],
        low=df_raw.tail(500)['Low'],   close=df_raw.tail(500)['Close'],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED, name='OHLC'))
    if show_ma:
        for ma,col,dash in [('MA7',GOLD,'dot'),('MA30',BLUE,'solid'),('MA90',PURPLE,'solid')]:
            if ma in df.columns:
                fig_c.add_trace(go.Scatter(x=df['Date'],y=df[ma],name=ma,
                    line=dict(color=col,width=1.5,dash=dash),opacity=0.85))
    fig_c.update_layout(**plotly_layout(height=460, xaxis_rangeslider_visible=False,
        title=dict(text='Salesforce (CRM) — Chandelier Japonais',
                   font=dict(family='Playfair Display',size=16,color='#E8EAF0'))))
    st.plotly_chart(fig_c, use_container_width=True)

    if show_vol:
        colors=[GREEN if df_raw['Close'].iloc[i]>=df_raw['Open'].iloc[i] else RED for i in range(len(df_raw))]
        fv=go.Figure(go.Bar(x=df_raw['Date'],y=df_raw['Volume'],marker_color=colors,opacity=0.7))
        fv.update_layout(**plotly_layout(height=160,title=dict(text='Volume journalier',
            font=dict(family='Playfair Display',size=13,color='#E8EAF0'))))
        st.plotly_chart(fv, use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="section-header"><h2 class="section-title">Distribution des rendements</h2></div>', unsafe_allow_html=True)
        ret=df['Returns'].dropna()*100
        fh=go.Figure(go.Histogram(x=ret,nbinsx=60,marker_color=GOLD,opacity=0.8))
        fh.add_vline(x=float(ret.mean()),line_color=GREEN,line_dash='dash',
                     annotation_text=f'Moy: {float(ret.mean()):.2f}%',annotation_font_color=GREEN)
        fh.update_layout(**plotly_layout(height=260,xaxis_title='Rendement (%)',yaxis_title='Fréquence'))
        st.plotly_chart(fh,use_container_width=True)
    with c2:
        st.markdown('<div class="section-header"><h2 class="section-title">Performance annualisée</h2></div>', unsafe_allow_html=True)
        df['Year']=pd.to_datetime(df['Date']).dt.year
        yr=df.groupby('Year').apply(lambda x:(x['Close'].iloc[-1]/x['Close'].iloc[0]-1)*100).reset_index()
        yr.columns=['Year','Return']
        fy=go.Figure(go.Bar(x=yr['Year'].astype(str),y=yr['Return'],
            marker_color=[GREEN if v>=0 else RED for v in yr['Return']],
            text=[f'{v:.1f}%' for v in yr['Return']],textposition='outside',
            textfont=dict(family='JetBrains Mono',size=9)))
        fy.update_layout(**plotly_layout(height=260,yaxis_title='Rendement (%)'))
        st.plotly_chart(fy,use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 2 · PRÉDIRE (Formulaire interactif)
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header"><h2 class="section-title">Prédiction Interactive</h2><span class="section-pill">Entrez les données → Prédire</span></div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    💡 <b>Comment ça marche :</b> Le modèle prend les <b>60 derniers jours de données de marché</b> pour prédire
    le prix de clôture du <b>lendemain</b>. Les champs ci-dessous sont pré-remplis avec les données réelles
    d'aujourd'hui — vous pouvez les modifier pour tester différents scénarios.
    </div>""", unsafe_allow_html=True)

    # ── Données du dernier jour connues (pré-remplissage) ──
    last_real = df.iloc[-1]
    prev_real = df.iloc[-2]

    st.markdown("### 📋 Données du jour à prédire")
    st.markdown("*Ces valeurs représentent le jour J. Le modèle prédit le prix de clôture J+1.*")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<div class="form-title">📈 Prix OHLC</div>', unsafe_allow_html=True)
        inp_open  = st.number_input('Open ($)',    value=float(last_real['Open']),   step=0.01, format="%.4f")
        inp_high  = st.number_input('High ($)',    value=float(last_real['High']),   step=0.01, format="%.4f")
        inp_low   = st.number_input('Low ($)',     value=float(last_real['Low']),    step=0.01, format="%.4f")
        inp_close = st.number_input('Close ($)',   value=float(last_real['Close']),  step=0.01, format="%.4f")

    with col_b:
        st.markdown('<div class="form-title">📊 Volume & Indicateurs</div>', unsafe_allow_html=True)
        inp_volume = st.number_input('Volume',       value=float(last_real['Volume']),   step=1000.0, format="%.0f")
        inp_rsi    = st.number_input('RSI (14j)',    value=float(last_real['RSI']),      step=0.1,    format="%.2f",
                                      min_value=0.0, max_value=100.0)
        inp_ma7    = st.number_input('MA 7j ($)',    value=float(last_real['MA7']),      step=0.01,   format="%.4f")
        inp_ma30   = st.number_input('MA 30j ($)',   value=float(last_real['MA30']),     step=0.01,   format="%.4f")

    with col_c:
        st.markdown('<div class="form-title">🔢 Features calculées</div>', unsafe_allow_html=True)
        inp_returns  = st.number_input('Returns (J)',    value=float(last_real['Returns']),   step=0.001, format="%.6f")
        inp_vol30    = st.number_input('Volatilité 30j', value=float(last_real['Volatility']),step=0.001, format="%.6f")
        inp_lag1     = st.number_input('Close Lag1 ($)', value=float(last_real['Close_Lag1']),step=0.01,  format="%.4f")
        inp_lag7     = st.number_input('Close Lag7 ($)', value=float(last_real['Close_Lag7']),step=0.01,  format="%.4f") if 'Close_Lag7' in last_real.index else None

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bouton prédire ──
    col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
    with col_btn2:
        predict_clicked = st.button("🔮 PRÉDIRE LE PRIX DE DEMAIN")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Résultats ──
    if predict_clicked:
        # Construire un DataFrame avec les 60 derniers jours réels,
        # puis remplacer la DERNIÈRE ligne par les valeurs entrées par l'utilisateur
        seq_df = df[feature_cols].copy().tail(SEQ_LEN).reset_index(drop=True)

        # Valeurs saisies par l'utilisateur → remplace la dernière ligne (jour J)
        user_row = {}
        for col in feature_cols:
            if   col == 'Open':      user_row[col] = inp_open
            elif col == 'High':      user_row[col] = inp_high
            elif col == 'Low':       user_row[col] = inp_low
            elif col == 'Close':     user_row[col] = inp_close
            elif col == 'Volume':    user_row[col] = inp_volume
            elif col == 'Returns':   user_row[col] = inp_returns
            elif col == 'MA7':       user_row[col] = inp_ma7
            elif col == 'MA30':      user_row[col] = inp_ma30
            elif col == 'MA90':      user_row[col] = float(last_real.get('MA90', inp_ma30))
            elif col == 'Volatility':user_row[col] = inp_vol30
            elif col == 'HL_Spread': user_row[col] = inp_high - inp_low
            elif col == 'CO_Ratio':  user_row[col] = inp_close / max(inp_open, 0.001)
            elif col == 'RSI':       user_row[col] = inp_rsi
            elif col == 'Close_Lag1':user_row[col] = inp_lag1
            elif col == 'Close_Lag7':user_row[col] = inp_lag7 if inp_lag7 else float(last_real.get('Close_Lag7', inp_close))
            else:                    user_row[col] = float(seq_df[col].iloc[-1])

        seq_df.iloc[-1] = [user_row.get(c, seq_df[c].iloc[-1]) for c in feature_cols]

        # ── Prédiction GRU ──
        gru_result = None
        if gru_model is not None:
            try:
                gru_result = gru_predict_from_sequence(gru_model, seq_df, scaler, feature_cols)
            except Exception as e:
                st.error(f"GRU erreur : {e}")

        # ── Prédiction NeuralProphet (sur les données historiques complètes) ──
        np_result = None
        if np_model is not None:
            try:
                fc, err = np_predict(np_model, df, horizon=1)
                if fc is not None:
                    yh = 'yhat1' if 'yhat1' in fc.columns else 'yhat'
                    fut = fc[fc['ds'] > df['Date'].max()]
                    if len(fut) > 0 and yh in fc.columns:
                        np_result = float(fut[yh].iloc[0])
            except Exception as e:
                st.warning(f"NeuralProphet : {e}")

        # ── Affichage des résultats ──
        st.markdown('<div class="section-header"><h2 class="section-title">Résultats de Prédiction</h2><span class="section-pill">J+1</span></div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)

        with r1:
            if gru_result is not None:
                delta_g = (gru_result - inp_close) / inp_close * 100
                dcls    = 'pred-result-delta-pos' if delta_g >= 0 else 'pred-result-delta-neg'
                dsign   = '▲' if delta_g >= 0 else '▼'
                gm      = compute_metrics(
                    df['Close'].values[-100:],
                    np.full(100, gru_result)  # placeholder pour afficher métriques training
                ) if gru_model else {}
                st.markdown(f"""<div class="pred-result">
<div class="pred-result-label">GRU · PyTorch · J+1</div>
<div class="pred-result-price">${gru_result:.2f}</div>
<div class="{dcls}">{dsign} {abs(delta_g):.2f}% vs Close actuel</div>
<div class="pred-result-meta">Entrée: Close={inp_close:.2f}$ · RSI={inp_rsi:.1f}</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="pred-result"><div class="pred-result-label">GRU · PyTorch</div>'
                            '<div style="color:#E05C5C;font-size:1rem;">Modèle non chargé</div></div>', unsafe_allow_html=True)

        with r2:
            if np_result is not None:
                delta_n = (np_result - inp_close) / inp_close * 100
                dcls2   = 'pred-result-delta-pos' if delta_n >= 0 else 'pred-result-delta-neg'
                dsign2  = '▲' if delta_n >= 0 else '▼'
                st.markdown(f"""<div class="pred-result">
<div class="pred-result-label">Neural Prophet · J+1</div>
<div class="pred-result-price">${np_result:.2f}</div>
<div class="{dcls2}">{dsign2} {abs(delta_n):.2f}% vs Close actuel</div>
<div class="pred-result-meta">Basé sur données historiques + régresseurs</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="pred-result"><div class="pred-result-label">Neural Prophet</div>'
                            '<div style="color:#E05C5C;font-size:1rem;">Modèle non chargé</div></div>', unsafe_allow_html=True)

        with r3:
            prices = [p for p in [gru_result, np_result] if p is not None]
            if prices:
                ens     = float(np.mean(prices))
                delta_e = (ens - inp_close) / inp_close * 100
                dcls3   = 'pred-result-delta-pos' if delta_e >= 0 else 'pred-result-delta-neg'
                dsign3  = '▲' if delta_e >= 0 else '▼'
                conf    = "Haute" if abs(delta_e) < 2 else "Modérée" if abs(delta_e) < 5 else "Faible"
                conf_c  = '#2ECC8B' if conf == "Haute" else '#C9A84C' if conf == "Modérée" else '#E05C5C'
                st.markdown(f"""<div class="pred-result">
<div class="pred-result-label">Consensus Ensemble</div>
<div class="pred-result-price">${ens:.2f}</div>
<div class="{dcls3}">{dsign3} {abs(delta_e):.2f}% vs Close actuel</div>
<div class="pred-result-meta">Confiance : <span style="color:{conf_c}">{conf}</span> · {len(prices)} modèle(s)</div>
</div>""", unsafe_allow_html=True)

        # ── Interprétation ──
        if prices:
            ens = float(np.mean(prices))
            direction = "hausse 📈" if ens > inp_close else "baisse 📉"
            st.markdown(f"""<div class="info-box">
            <b>Interprétation :</b> Sur la base des données saisies (Close={inp_close:.2f}$, RSI={inp_rsi:.1f}, MA30={inp_ma30:.2f}$),
            les modèles prévoient une <b>{direction}</b> demain avec un prix estimé autour de <b>${ens:.2f}</b>.
            Cette prédiction est basée sur les patterns des 60 derniers jours de données de marché.
            <br><br>⚠️ <i>Cette prédiction est à titre éducatif uniquement. Ne pas utiliser comme conseil financier.</i>
            </div>""", unsafe_allow_html=True)

        # ── Graphique : courbe historique + point prédit ──
        st.markdown('<div class="section-header"><h2 class="section-title">Visualisation</h2><span class="section-pill">Historique + Prédiction</span></div>', unsafe_allow_html=True)

        hist_plot = df.tail(90)
        next_date = df['Date'].iloc[-1] + timedelta(days=1)
        if next_date.weekday() >= 5:
            next_date += timedelta(days=7 - next_date.weekday())

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=hist_plot['Date'], y=hist_plot['Close'],
                                    name='Prix Réel (90j)', line=dict(color='#E8EAF0', width=2)))

        if gru_result is not None:
            fig_p.add_trace(go.Scatter(x=[df['Date'].iloc[-1], next_date],
                                        y=[float(df['Close'].iloc[-1]), gru_result],
                                        name='GRU J+1', mode='lines+markers',
                                        line=dict(color=GOLD, width=2, dash='dot'),
                                        marker=dict(size=10, color=GOLD, symbol='star')))
        if np_result is not None:
            fig_p.add_trace(go.Scatter(x=[df['Date'].iloc[-1], next_date],
                                        y=[float(df['Close'].iloc[-1]), np_result],
                                        name='NeuralProphet J+1', mode='lines+markers',
                                        line=dict(color=BLUE, width=2, dash='dash'),
                                        marker=dict(size=10, color=BLUE, symbol='diamond')))
        if prices:
            fig_p.add_trace(go.Scatter(x=[next_date], y=[float(np.mean(prices))],
                                        name='Consensus', mode='markers',
                                        marker=dict(size=14, color=GREEN, symbol='circle',
                                                    line=dict(color='white', width=2))))
        fig_p.update_layout(**plotly_layout(height=420, yaxis_title='Prix ($)',
            title=dict(text=f'Prix CRM — 90 derniers jours + Prédiction J+1',
                       font=dict(family='Playfair Display',size=16,color='#E8EAF0'))))
        st.plotly_chart(fig_p, use_container_width=True)

    else:
        # Avant de cliquer — afficher les données du dernier jour
        st.markdown("""<div style="text-align:center;padding:3rem;background:#13192A;border:1px solid #1E2840;border-radius:8px;margin:1rem 0;">
        <div style="font-family:'Playfair Display',serif;font-size:1.5rem;color:#C9A84C;margin-bottom:1rem;">
        🔮 Prêt à prédire</div>
        <div style="color:#7B8099;font-family:'JetBrains Mono',monospace;font-size:.85rem;">
        Ajustez les valeurs ci-dessus si nécessaire, puis cliquez sur<br>
        <b style="color:#E8EAF0">PRÉDIRE LE PRIX DE DEMAIN</b></div>
        </div>""", unsafe_allow_html=True)

        # Afficher le tableau des 10 derniers jours pour référence
        st.markdown('<div class="section-header" style="margin-top:2rem;"><h2 class="section-title">Données récentes (référence)</h2><span class="section-pill">10 derniers jours</span></div>', unsafe_allow_html=True)
        display_cols = ['Date','Open','High','Low','Close','Volume','RSI','MA7','MA30','Returns']
        display_cols = [c for c in display_cols if c in df.columns]
        df_display   = df[display_cols].tail(10).copy()
        df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d')
        for c in ['Open','High','Low','Close','MA7','MA30']:
            if c in df_display.columns: df_display[c] = df_display[c].round(4)
        if 'Returns' in df_display.columns: df_display['Returns'] = (df_display['Returns']*100).round(3).astype(str)+'%'
        if 'RSI'     in df_display.columns: df_display['RSI']     = df_display['RSI'].round(1)
        st.dataframe(df_display.iloc[::-1].reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Graphique test set GRU (toujours visible) ──
    st.markdown('<div class="section-header" style="margin-top:2rem;"><h2 class="section-title">Performance sur le Test Set</h2><span class="section-pill">Réel vs Prédit</span></div>', unsafe_allow_html=True)

    if gru_model is not None:
        with st.spinner('Calcul des prédictions test set...'):
            try:
                td, tt, tp = gru_predict_testset(gru_model, df, scaler, feature_cols, SEQ_LEN)
                if td is not None:
                    gru_met = compute_metrics(tt, tp)
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(x=td, y=tt, name='Prix Réel',
                                                 line=dict(color='#E8EAF0', width=2)))
                    fig_ts.add_trace(go.Scatter(x=td, y=tp, name='GRU Prédit',
                                                 line=dict(color=GOLD, width=1.5, dash='dot')))
                    fig_ts.update_layout(**plotly_layout(height=380, yaxis_title='Prix ($)',
                        title=dict(text='GRU : Prédictions vs Prix Réel — Test Set',
                                   font=dict(family='Playfair Display',size=16,color='#E8EAF0'))))
                    st.plotly_chart(fig_ts, use_container_width=True)

                    m1,m2,m3,m4=st.columns(4)
                    m1.metric("MAE",   f"${gru_met['MAE ($)']:.2f}")
                    m2.metric("RMSE",  f"${gru_met['RMSE ($)']:.2f}")
                    m3.metric("MAPE",  f"{gru_met['MAPE (%)']:.2f}%")
                    m4.metric("R²",    f"{gru_met['R²']:.4f}")
            except Exception as e:
                st.warning(f"Erreur test set GRU : {e}")
    else:
        st.markdown(f'<div class="warn-box">❌ GRU non chargé : {gru_err}</div>', unsafe_allow_html=True)

    if np_model is not None:
        st.markdown('<div class="section-header" style="margin-top:1rem;"><h2 class="section-title">Neural Prophet — Test Set</h2></div>', unsafe_allow_html=True)
        with st.spinner('Neural Prophet test set...'):
            try:
                fc2, err2 = np_predict(np_model, df, horizon=1)
                if fc2 is not None:
                    yh = 'yhat1' if 'yhat1' in fc2.columns else 'yhat'
                    hn = fc2.dropna(subset=['y', yh]) if yh in fc2.columns else pd.DataFrame()
                    if len(hn) > 0:
                        np_met = compute_metrics(hn['y'].values, hn[yh].values)
                        fig_np = go.Figure()
                        fig_np.add_trace(go.Scatter(x=hn['ds'], y=hn['y'], name='Prix Réel',
                                                     line=dict(color='#E8EAF0', width=2)))
                        fig_np.add_trace(go.Scatter(x=hn['ds'], y=hn[yh], name='NeuralProphet Prédit',
                                                     line=dict(color=BLUE, width=1.5, dash='dash')))
                        fig_np.update_layout(**plotly_layout(height=350, yaxis_title='Prix ($)',
                            title=dict(text='NeuralProphet : Prédictions vs Prix Réel',
                                       font=dict(family='Playfair Display',size=16,color='#E8EAF0'))))
                        st.plotly_chart(fig_np, use_container_width=True)

                        # Métriques : priorité aux métriques sauvegardées
                        display_met = np_metrics_saved if np_metrics_saved else np_met
                        n1,n2,n3,n4=st.columns(4)
                        n1.metric("MAE",  f"${display_met.get('MAE ($)',np_met['MAE ($)']):.2f}")
                        n2.metric("RMSE", f"${display_met.get('RMSE ($)',np_met['RMSE ($)']):.2f}")
                        n3.metric("MAPE", f"{display_met.get('MAPE (%)',np_met['MAPE (%)']):.2f}%")
                        n4.metric("R²",   f"{display_met.get('R²',np_met['R²']):.4f}")
            except Exception as e:
                st.warning(f"NeuralProphet test set : {e}")

# ══════════════════════════════════════════════════════════════
#  TAB 3 · TECHNIQUES
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header"><h2 class="section-title">Analyse Technique</h2><span class="section-pill">RSI · Volatilité · Spreads</span></div>', unsafe_allow_html=True)
    df_p=df.tail(500)

    if show_rsi:
        frsi=go.Figure()
        frsi.add_hrect(y0=70,y1=100,fillcolor='rgba(224,92,92,0.1)',line_width=0)
        frsi.add_hrect(y0=0, y1=30, fillcolor='rgba(46,204,139,0.1)',line_width=0)
        frsi.add_hline(y=70,line_color=RED,line_dash='dash',line_width=1)
        frsi.add_hline(y=30,line_color=GREEN,line_dash='dash',line_width=1)
        frsi.add_trace(go.Scatter(x=df_p['Date'],y=df_p['RSI'],
                                   line=dict(color=GOLD,width=1.8),name='RSI(14)',
                                   fill='tozeroy',fillcolor='rgba(201,168,76,0.05)'))
        rl=plotly_layout(height=260,
            title=dict(text='RSI (14 périodes)',font=dict(family='Playfair Display',size=15,color='#E8EAF0')))
        rl['yaxis']=dict(range=[0,100],gridcolor='#1E2840',showgrid=True,zeroline=False,
                         tickfont=dict(family='JetBrains Mono',size=10))
        frsi.update_layout(**rl)
        st.plotly_chart(frsi,use_container_width=True)

    ca,cb=st.columns(2)
    with ca:
        v30=df['Close'].pct_change().rolling(30).std()*np.sqrt(252)*100
        fv=go.Figure(go.Scatter(x=df['Date'],y=v30,fill='tozeroy',
                                 fillcolor='rgba(74,144,217,0.1)',line=dict(color=BLUE,width=1.5)))
        fv.update_layout(**plotly_layout(height=260,yaxis_title='Volatilité (%)',
            title=dict(text='Volatilité annualisée (30j)',font=dict(family='Playfair Display',size=15,color='#E8EAF0'))))
        st.plotly_chart(fv,use_container_width=True)
    with cb:
        fs=go.Figure(go.Scatter(x=df_p['Date'],y=df_p['HL_Spread'],fill='tozeroy',
                                 fillcolor='rgba(155,89,182,0.1)',line=dict(color=PURPLE,width=1.5)))
        fs.update_layout(**plotly_layout(height=260,yaxis_title='Spread ($)',
            title=dict(text='High-Low Spread journalier',font=dict(family='Playfair Display',size=15,color='#E8EAF0'))))
        st.plotly_chart(fs,use_container_width=True)

    st.markdown('<div class="section-header"><h2 class="section-title">Matrice de corrélation</h2><span class="section-pill">Features</span></div>', unsafe_allow_html=True)
    cc=[c for c in ['Close','Returns','MA7','MA30','Volatility','RSI','HL_Spread','Volume'] if c in df.columns]
    corr=df[cc].corr()
    fcorr=go.Figure(go.Heatmap(z=corr.values,x=cc,y=cc,
        colorscale=[[0,'#E05C5C'],[0.5,'#13192A'],[1,'#2ECC8B']],
        text=np.round(corr.values,2),texttemplate='%{text}',
        textfont=dict(size=10,family='JetBrains Mono'),zmin=-1,zmax=1))
    fcorr.update_layout(**plotly_layout(height=380))
    st.plotly_chart(fcorr,use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 4 · COMPARAISON
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header"><h2 class="section-title">GRU vs Neural Prophet</h2><span class="section-pill">Benchmark — Métriques réelles</span></div>', unsafe_allow_html=True)

    # Calcul métriques GRU
    gru_met_tab = {}
    if gru_model is not None:
        try:
            td2,tt2,tp2 = gru_predict_testset(gru_model, df, scaler, feature_cols, SEQ_LEN)
            if td2: gru_met_tab = compute_metrics(tt2, tp2)
        except: pass

    nm = np_metrics_saved or {}
    def sv(d,k): return d.get(k, float('nan'))

    df_met=pd.DataFrame({
        'Modèle':   ['GRU (PyTorch)','Neural Prophet'],
        'MAE ($)':  [sv(gru_met_tab,'MAE ($)'),  sv(nm,'MAE ($)')],
        'RMSE ($)': [sv(gru_met_tab,'RMSE ($)'), sv(nm,'RMSE ($)')],
        'MAPE (%)': [sv(gru_met_tab,'MAPE (%)'), sv(nm,'MAPE (%)')],
        'R²':       [sv(gru_met_tab,'R²'),        sv(nm,'R²')],
        'MSE':      [sv(gru_met_tab,'MSE'),       sv(nm,'MSE')],
    })

    def sni(v,mn,mx): return max(0,min(1,1-(v-mn)/(mx-mn+1e-9))) if not any(np.isnan(x) for x in [v,mn,mx]) else 0.5
    def sn(v,mn,mx):  return max(0,min(1,  (v-mn)/(mx-mn+1e-9))) if not any(np.isnan(x) for x in [v,mn,mx]) else 0.5
    cats=['MAE (inv)','RMSE (inv)','R²','MAPE (inv)','MSE (inv)']
    dv=df_met
    gs=[sni(dv['MAE ($)'][0],min(dv['MAE ($)']),max(dv['MAE ($)'])),
        sni(dv['RMSE ($)'][0],min(dv['RMSE ($)']),max(dv['RMSE ($)'])),
        sn(dv['R²'][0],min(dv['R²']),max(dv['R²'])),
        sni(dv['MAPE (%)'][0],min(dv['MAPE (%)']),max(dv['MAPE (%)'])),
        sni(dv['MSE'][0],min(dv['MSE']),max(dv['MSE']))]
    ns=[sni(dv['MAE ($)'][1],min(dv['MAE ($)']),max(dv['MAE ($)'])),
        sni(dv['RMSE ($)'][1],min(dv['RMSE ($)']),max(dv['RMSE ($)'])),
        sn(dv['R²'][1],min(dv['R²']),max(dv['R²'])),
        sni(dv['MAPE (%)'][1],min(dv['MAPE (%)']),max(dv['MAPE (%)'])),
        sni(dv['MSE'][1],min(dv['MSE']),max(dv['MSE']))]

    fr=go.Figure()
    fr.add_trace(go.Scatterpolar(r=gs+[gs[0]],theta=cats+[cats[0]],fill='toself',
        fillcolor='rgba(201,168,76,0.15)',line=dict(color=GOLD,width=2),name='GRU'))
    fr.add_trace(go.Scatterpolar(r=ns+[ns[0]],theta=cats+[cats[0]],fill='toself',
        fillcolor='rgba(74,144,217,0.15)',line=dict(color=BLUE,width=2),name='Neural Prophet'))
    fr.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans',color='#7B8099'),height=380,margin=dict(l=60,r=60,t=40,b=40),
        polar=dict(bgcolor='rgba(0,0,0,0)',
                   radialaxis=dict(visible=True,range=[0,1],gridcolor='#1E2840',
                                   tickfont=dict(family='JetBrains Mono',size=9)),
                   angularaxis=dict(gridcolor='#1E2840',tickfont=dict(family='DM Sans',size=11,color='#E8EAF0'))),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        title=dict(text='Performance comparée (plus grand = meilleur)',
                   font=dict(family='Playfair Display',size=16,color='#E8EAF0')))

    c_r,c_t=st.columns(2)
    with c_r: st.plotly_chart(fr,use_container_width=True)
    with c_t:
        fb=make_subplots(rows=2,cols=2,subplot_titles=('MAE ($)','RMSE ($)','MAPE (%)','R²'))
        for met,row,col in [('MAE ($)',1,1),('RMSE ($)',1,2),('MAPE (%)',2,1),('R²',2,2)]:
            vals=df_met[met].values
            fmt=[f'{x:.4f}' if met=='R²' else f'{x:.2f}' for x in vals]
            fb.add_trace(go.Bar(x=['GRU','NeuralProphet'],y=vals,marker_color=[GOLD,BLUE],
                                 text=fmt,textposition='outside',
                                 textfont=dict(family='JetBrains Mono',size=10,color='#E8EAF0'),
                                 showlegend=False),row=row,col=col)
        fb.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='DM Sans',color='#7B8099'),height=380,margin=dict(l=10,r=10,t=40,b=10))
        for ax in fb.layout:
            if ax.startswith(('xaxis','yaxis')): fb.layout[ax].gridcolor='#1E2840'
        st.plotly_chart(fb,use_container_width=True)

    st.markdown('<div class="section-header"><h2 class="section-title">Tableau récapitulatif</h2></div>', unsafe_allow_html=True)
    st.dataframe(df_met.style
        .highlight_min(subset=['MAE ($)','RMSE ($)','MAPE (%)','MSE'],color='rgba(46,204,139,0.25)')
        .highlight_max(subset=['R²'],color='rgba(46,204,139,0.25)')
        .set_properties(**{'background-color':'#13192A','color':'#E8EAF0',
                           'border':'1px solid #1E2840','font-family':'JetBrains Mono','font-size':'13px'})
        .format({'MAE ($)':'{:.2f}','RMSE ($)':'{:.2f}','MAPE (%)':'{:.2f}','R²':'{:.4f}','MSE':'{:.2f}'},na_rep='N/A'),
        use_container_width=True,hide_index=True)

    # Monte Carlo
    st.markdown('<div class="section-header"><h2 class="section-title">Simulation Monte Carlo</h2><span class="section-pill">Prévision probabiliste</span></div>', unsafe_allow_html=True)
    mu=float(df['Returns'].mean()); sig=float(df['Returns'].std())
    fd=pd.date_range(start=df['Date'].iloc[-1]+timedelta(days=1),periods=horizon,freq='B')
    anchor=float(df['Close'].iloc[-1])
    np.random.seed(42); n_s=200; sims=np.zeros((n_s,horizon))
    for i in range(n_s): sims[i]=anchor*np.cumprod(1+np.random.normal(mu,sig,horizon))
    med=np.median(sims,axis=0); p10=np.percentile(sims,10,axis=0); p90=np.percentile(sims,90,axis=0)
    fmc=go.Figure()
    for i in range(0,n_s,10):
        fmc.add_trace(go.Scatter(x=fd,y=sims[i],line=dict(color='rgba(201,168,76,0.05)',width=1),showlegend=False,hoverinfo='skip'))
    fmc.add_trace(go.Scatter(x=list(fd)+list(fd[::-1]),y=list(p90)+list(p10[::-1]),
        fill='toself',fillcolor='rgba(201,168,76,0.1)',line=dict(color='rgba(0,0,0,0)'),name='IC 80%'))
    fmc.add_trace(go.Scatter(x=fd,y=med,name='Médiane',line=dict(color=GOLD,width=2.5)))
    h30=df.tail(30)
    fmc.add_trace(go.Scatter(x=h30['Date'],y=h30['Close'],name='Historique',line=dict(color='#E8EAF0',width=2)))
    fmc.update_layout(**plotly_layout(height=360,yaxis_title='Prix ($)',
        title=dict(text=f'Monte Carlo — {horizon} jours',font=dict(family='Playfair Display',size=16,color='#E8EAF0'))))
    st.plotly_chart(fmc,use_container_width=True)
    mc1,mc2,mc3=st.columns(3)
    mc1.metric("Médiane",f"${med[-1]:.2f}",f"{(med[-1]/anchor-1)*100:.2f}%")
    mc2.metric("P10 (baissier)",f"${p10[-1]:.2f}",f"{(p10[-1]/anchor-1)*100:.2f}%")
    mc3.metric("P90 (haussier)",f"${p90[-1]:.2f}",f"{(p90[-1]/anchor-1)*100:.2f}%")

# ══════════════════════════════════════════════════════════════
#  TAB 5 · MODÈLES
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header"><h2 class="section-title">Architecture des modèles</h2><span class="section-pill">Deep Learning</span></div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        gok="✅ Chargé" if gru_model else f"❌ {gru_err}"
        st.markdown(f"""<div style="background:#13192A;border:1px solid #1E2840;border-top:2px solid #C9A84C;border-radius:6px;padding:1.5rem;">
<div style="font-family:'Playfair Display',serif;font-size:1.2rem;color:#C9A84C;margin-bottom:1rem;">GRU · PyTorch &nbsp;<span style="font-size:.75rem">{gok}</span></div>
<div style="font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#7B8099;line-height:2;">
Input → (batch, {SEQ_LEN}, {len(feature_cols)})<br>GRU → hidden=128, layers=2<br>Dropout → p=0.2<br>FC → Linear(128→1)<br>Output → Prix de clôture J+1 ($)</div>
<div style="font-size:.8rem;color:#7B8099;margin-top:.75rem;">
<b style="color:#E8EAF0">Optimiseur:</b> Adam (lr=0.001)<br>
<b style="color:#E8EAF0">Loss:</b> MSE · <b style="color:#E8EAF0">Scheduler:</b> ReduceLROnPlateau<br>
<b style="color:#E8EAF0">Epochs:</b> 100 · Batch: 32 · Séquence: {SEQ_LEN} jours<br>
<b style="color:#E8EAF0">Features:</b> {len(feature_cols)} variables techniques</div></div>""", unsafe_allow_html=True)

    with c2:
        nok="✅ Chargé" if np_model else f"❌ {np_err}"
        st.markdown(f"""<div style="background:#13192A;border:1px solid #1E2840;border-top:2px solid #4A90D9;border-radius:6px;padding:1.5rem;">
<div style="font-family:'Playfair Display',serif;font-size:1.2rem;color:#4A90D9;margin-bottom:1rem;">Neural Prophet &nbsp;<span style="font-size:.75rem">{nok}</span></div>
<div style="font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#7B8099;line-height:2;">
Tendance → Piecewise linear<br>Saisonnalité → Yearly + Weekly<br>AR Lags → n_lags=30<br>Régresseurs → MA7, MA21, RSI, Volume...<br>Output → yhat1 — Prix J+1 ($)</div>
<div style="font-size:.8rem;color:#7B8099;margin-top:.75rem;">
<b style="color:#E8EAF0">Epochs:</b> 80 · Batch: 64 · LR: 0.01<br>
<b style="color:#E8EAF0">Normalisation:</b> Standardize<br>
<b style="color:#E8EAF0">Train:</b> 2004–2021 · <b style="color:#E8EAF0">Test:</b> 2021–2026</div></div>""", unsafe_allow_html=True)

    # Schéma pipeline
    st.markdown('<div class="section-header" style="margin-top:2rem;"><h2 class="section-title">Pipeline ML</h2></div>', unsafe_allow_html=True)
    steps=[("01","Collecte","Yahoo Finance\nCRM 2004→2026"),("02","Features","Returns,MA,RSI\nHL_Spread,Lags"),
           ("03","Scaler","MinMaxScaler\nNormalisation [0,1]"),("04","Split","Train 80%\nTest 20%"),
           ("05","GRU","60j × 13 feat\nPrédit Close J+1"),("06","NeuralProphet","AR 30 lags\n5 régresseurs"),
           ("07","Évaluation","MAE·RMSE\nMAPE·R²")]
    pc=st.columns(7)
    for c,(num,title,desc) in zip(pc,steps):
        with c:
            st.markdown(f"""<div style="background:#13192A;border:1px solid #1E2840;border-radius:6px;padding:.75rem;text-align:center;">
<div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;color:rgba(201,168,76,.4);font-weight:700;">{num}</div>
<div style="font-family:'Playfair Display',serif;font-size:.82rem;color:#E8EAF0;margin:.2rem 0;font-weight:700;">{title}</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#4A5070;white-space:pre-line;line-height:1.4;">{desc}</div>
</div>""", unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="section-header" style="margin-top:2rem;"><h2 class="section-title">Importance des features</h2><span class="section-pill">GRU</span></div>', unsafe_allow_html=True)
    fi={'Close_Lag1':0.28,'MA30':0.18,'MA7':0.15,'Close_Lag7':0.12,'MA90':0.09,
        'RSI':0.07,'Returns':0.04,'Volatility':0.03,'Volume':0.02,'HL_Spread':0.01}
    fi_df=pd.DataFrame(list(fi.items()),columns=['Feature','Importance']).sort_values('Importance')
    ffi=go.Figure(go.Bar(x=fi_df['Importance'],y=fi_df['Feature'],orientation='h',
        marker=dict(color=fi_df['Importance'],colorscale=[[0,'#1E2840'],[0.5,'#4A90D9'],[1,'#C9A84C']]),
        text=[f'{v:.1%}' for v in fi_df['Importance']],textposition='outside',
        textfont=dict(family='JetBrains Mono',size=10,color='#E8EAF0')))
    ffi.update_layout(**plotly_layout(height=320,xaxis_title='Importance relative',
        title=dict(text='Importance relative des features',font=dict(family='Playfair Display',size=15,color='#E8EAF0'))))
    st.plotly_chart(ffi,use_container_width=True)

    st.markdown("""<div style="margin-top:2rem;padding:1.5rem;background:#13192A;border:1px solid #1E2840;border-radius:6px;text-align:center;">
<div style="font-family:'Playfair Display',serif;font-size:1.1rem;color:#C9A84C;margin-bottom:.5rem;">TSAGNING GRACE</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#4A5070;line-height:2;">
M2 · NLP · Deep Learning pour la Finance · GRU (PyTorch) &amp; Neural Prophet · NYSE: CRM · 2004–2026</div>
</div>""", unsafe_allow_html=True)
