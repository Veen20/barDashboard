# dashboard_interaktif.py
# Minimalist Premium Sentiment Dashboard
# - Top navigation (minimal)
# - Background: samsat.jpg (in repo root) blurred + blue overlay
# - Real-time Google Sheets -> CSV export
# - Preprocessing Bahasa Indonesia (Sastrawi if available)
# - IndoBERT inference (mdhugol/indonesia-bert-sentiment-classification)
# - Fast mode (dummy labels) for quick demo
# - Clean layout: filters on top, metrics, chart, table, download
#
# Usage:
#   pip install -r requirements.txt
#   streamlit run dashboard_interaktif.py
# ============================================================

import io, os, re, base64
from typing import List, Tuple
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# NLP/model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional statsmodels for trendline (not required)
try:
    import statsmodels.api  # noqa
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# ------------------ CONFIG ------------------
GSHEET_ID = "1VL8FwJrAAZHqEDErlkhPtQ-a69O4JcRkOBmnYPKH2PY"
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/export?format=csv"

# Streamlit page
st.set_page_config(page_title="Samsat Sentiment — Minimal", page_icon="🧠", layout="wide")

# ------------------ STYLES (TOP NAV + BACKGROUND + GLASS CARDS) ------------------
# Load background image if exists in repo root as 'samsat.jpg'; else use gradient
bg_style = ""
if os.path.exists("samsat.jpg"):
    with open("samsat.jpg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    bg_style = f"background-image: url('data:image/jpeg;base64,{b64}');"
else:
    # fallback gradient
    bg_style = "background: linear-gradient(120deg,#3a1b48,#2e86c1);"

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
      html, body, .stApp {{ height:100%; }}
      .app-bg {{
        position: fixed; inset: 0; z-index:-99;
        {bg_style}
        background-size: cover; background-position: center;
        filter: blur(8px) brightness(.55);
        transform: scale(1.02);
      }}
      .overlay {{
        position: fixed; inset: 0; z-index:-90;
        background: linear-gradient(135deg, rgba(93,63,211,0.28), rgba(46,134,193,0.18));
        mix-blend-mode: multiply;
        pointer-events: none;
      }}
      .topbar {{
        display:flex; align-items:center; justify-content:space-between;
        padding:18px 28px; border-radius:10px;
        margin:18px 36px;
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.06);
        color: #EAF2FF;
      }}
      .brand {{ display:flex; gap:12px; align-items:center; }}
      .logo {{ width:56px; height:56px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-weight:800; color:white; background: linear-gradient(135deg,#5D3FD3,#2E86C1); box-shadow:0 8px 20px rgba(10,20,40,0.4); }}
      .title {{ font-family:Inter, sans-serif; font-weight:700; font-size:20px; color:#fff; margin:0; }}
      .subtitle {{ font-size:12px; color: rgba(235,245,255,0.85); margin:0; opacity:0.9; }}
      .controls{{ display:flex; gap:12px; align-items:center; }}
      .glass {{ background: rgba(255,255,255,0.06); border-radius:12px; padding:12px; border:1px solid rgba(255,255,255,0.06); backdrop-filter: blur(6px); color:#fff; }}
      .metric-title{ font-size:11px; color:rgba(255,255,255,0.8); text-transform:uppercase; letter-spacing:.06em; }
      .metric-value{ font-size:22px; font-weight:700; color:#fff; }
      /* dataframes: improve contrast */
      div[data-testid="stDataFrame"] table {{ background: rgba(255,255,255,0.02); color: #eaf2ff; }}
      .small-muted{{ color: rgba(235,245,255,0.75); font-size:12px; }}
      @media (max-width:768px) {{
        .topbar {{ margin:10px; padding:12px; flex-direction:column; align-items:flex-start; gap:8px; }}
        .controls {{ flex-wrap:wrap; }}
      }}
    </style>
    <div class="app-bg"></div>
    <div class="overlay"></div>
    """,
    unsafe_allow_html=True,
)

# ------------------ HELPERS: text cleaning ------------------
# Use Sastrawi stopwords if available, else empty set
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    factory = StopWordRemoverFactory()
    STOPWORDS = set(factory.get_stop_words())
except Exception:
    STOPWORDS = set()

def _compress_repeats(s: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1\1', s)

def clean_text(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'@[^\s]+|#[^\s]+', ' ', t)
    t = t.encode('ascii', 'ignore').decode('ascii')  # remove emojis
    t = re.sub(r'[^a-z\s]', ' ', t)
    t = _compress_repeats(t)
    if STOPWORDS:
        return " ".join([w for w in t.split() if w and w not in STOPWORDS])
    else:
        return " ".join([w for w in t.split() if w])

# ------------------ MODEL loader & inference (cached) ------------------
@st.cache_resource(show_spinner=True)
def load_model():
    PRETRAIN = "mdhugol/indonesia-bert-sentiment-classification"
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAIN)
    mapping = {"LABEL_0":"positif","LABEL_1":"netral","LABEL_2":"negatif"}
    return tokenizer, model, mapping

@st.cache_data(show_spinner=False)
def infer_from_bytes(csv_bytes: bytes, fast_mode: bool, batch_size: int = 32) -> pd.DataFrame:
    bio = io.BytesIO(csv_bytes)
    df = pd.read_csv(bio)
    df.columns = [c.strip() for c in df.columns]
    # map common names case-insensitively
    col_map = {}
    for needed in ["No","Tanggal","Komentar"]:
        for c in df.columns:
            if c.strip().lower() == needed.lower():
                col_map[c] = needed
    if col_map:
        df = df.rename(columns=col_map)
    if not set(["No","Tanggal","Komentar"]).issubset(set(df.columns)):
        return pd.DataFrame({"error":["Missing columns: No, Tanggal, Komentar"]})
    df = df.dropna(subset=["Komentar"]).reset_index(drop=True)
    df["Komentar_Bersih"] = df["Komentar"].astype(str).apply(clean_text)
    if fast_mode:
        rng = np.random.default_rng(2025)
        df["Sentimen"] = rng.choice(["positif","netral","negatif"], size=len(df))
        df["Kepercayaan"] = (rng.random(len(df))*0.4 + 0.6).round(4)
        return df
    tokenizer, model, mapping = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    texts = df["Komentar_Bersih"].astype(str).tolist()
    labels, confs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=160, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc).logits
            probs = torch.softmax(out, dim=-1)
            topv, topi = probs.max(dim=1)
        for idx, p in zip(topi.cpu().tolist(), topv.cpu().tolist()):
            labels.append(mapping.get(f"LABEL_{idx}", "netral"))
            confs.append(float(p))
    df["Sentimen"] = labels
    df["Kepercayaan"] = np.round(confs, 4)
    return df

# ------------------ Top controls (filters) displayed inline in topbar ------------------
# We'll implement controls in the main app header area (not left sidebar)
st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="logo">SP</div>
        <div>
          <div class="title">Samsat Sentiment</div>
          <div class="subtitle">Realtime • IndoBERT • Minimal modern UI</div>
        </div>
      </div>
      <div class="controls" id="controls-area"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Build controls area using normal Streamlit layout just under the topbar to keep alignment
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2,2,1,1])
with ctrl1:
    # date range filter defaults to full if Tanggal exists
    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
with ctrl2:
    st.markdown("", unsafe_allow_html=True)
with ctrl3:
    fast_mode = st.checkbox("Fast mode (no model) — demo cepat", value=False)
with ctrl4:
    sample_limit = st.selectbox("Limit rows", options=[100, 250, 500, 1000, 2000], index=3)

# Now load data bytes (Google Sheets or optional file upload)
colA, colB = st.columns([1,1])
with colA:
    mode = st.radio("Sumber data:", ["Google Sheets (real-time)", "Upload CSV/XLSX"], horizontal=True)
with colB:
    uploaded = st.file_uploader("Jika upload, pilih file", type=["csv","xlsx"])

# fetch CSV bytes helper (cached briefly)
@st.cache_data(ttl=45)
def fetch_csv_bytes(url: str) -> bytes:
    dfx = pd.read_csv(url)
    bio = io.BytesIO(); dfx.to_csv(bio, index=False)
    return bio.getvalue()

# determine raw_bytes
if mode.startswith("Google"):
    try:
        raw_bytes = fetch_csv_bytes(GSHEET_CSV_URL)
    except Exception as e:
        st.error("Gagal membaca Google Sheets: " + str(e))
        st.stop()
else:
    if uploaded is None:
        st.info("Silakan upload file CSV/XLSX atau pilih Google Sheets.")
        st.stop()
    else:
        if uploaded.name.lower().endswith(".xlsx"):
            df_local = pd.read_excel(uploaded)
            b = io.BytesIO(); df_local.to_csv(b, index=False); raw_bytes = b.getvalue()
        else:
            raw_bytes = uploaded.read()

# run inference (cached)
with st.spinner("Memproses & menganalisis..."):
    df_all = infer_from_bytes(raw_bytes, fast_mode=fast_mode, batch_size=32)

if "error" in df_all.columns:
    st.error(df_all["error"].iat[0])
    st.stop()

# parse date column
df_all.columns = [c.strip() for c in df_all.columns]
if "Tanggal" in df_all.columns:
    df_all["Tanggal"] = pd.to_datetime(df_all["Tanggal"], errors="coerce").dt.date

# apply sample limit and reorder columns
df = df_all.head(min(len(df_all), int(sample_limit))).copy().reset_index(drop=True)

# Top quick stats row (compact glass cards)
c1, c2, c3, c4 = st.columns([1,1,1,1], gap="small")
with c1:
    st.markdown(f"<div class='glass'><div class='metric-title'>Total</div><div class='metric-value'>{len(df):,}</div></div>", unsafe_allow_html=True)
with c2:
    p = (df['Sentimen']=='positif').sum()
    st.markdown(f"<div class='glass'><div class='metric-title'>Positif</div><div class='metric-value'>{p:,}</div></div>", unsafe_allow_html=True)
with c3:
    n = (df['Sentimen']=='netral').sum()
    st.markdown(f"<div class='glass'><div class='metric-title'>Netral</div><div class='metric-value'>{n:,}</div></div>", unsafe_allow_html=True)
with c4:
    neg = (df['Sentimen']=='negatif').sum()
    st.markdown(f"<div class='glass'><div class='metric-title'>Negatif</div><div class='metric-value'>{neg:,}</div></div>", unsafe_allow_html=True)

# Insight short
dominant = "Positif" if p>=max(n,neg) else ("Netral" if n>=neg else "Negatif")
avg_conf = float(df["Kepercayaan"].mean()) if "Kepercayaan" in df.columns else 0.0
st.markdown(f"<div style='margin-top:8px;'><span class='small-muted'>Insight: Dominan <b>{dominant}</b> • Rata-rata confidence: <b>{avg_conf*100:.1f}%</b></span></div>", unsafe_allow_html=True)

st.markdown("---")

# Main visualization row: pie + trend (if date available)
left, right = st.columns([1.3,1])
with left:
    dist = df["Sentimen"].value_counts().rename_axis("Sentimen").reset_index(name="Jumlah")
    fig_pie = px.pie(dist, names="Sentimen", values="Jumlah", hole=.35,
                     color_discrete_map={"positif":"#2ECC71","netral":"#F4D03F","negatif":"#E74C3C"})
    fig_pie.update_layout(margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_pie, use_container_width=True, theme="streamlit")
with right:
    if "Tanggal" in df.columns:
        tmp = df.groupby([df["Tanggal"].astype(str), "Sentimen"]).size().reset_index(name="Jumlah")
        tmp.columns = ["Tanggal","Sentimen","Jumlah"]
        fig_line = px.line(tmp, x="Tanggal", y="Jumlah", color="Sentimen", markers=True)
        fig_line.update_layout(margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Kolom Tanggal tidak tersedia — trend per-hari tidak bisa ditampilkan.")

st.markdown("---")

# Wordcloud (compact) and top words
w1, w2 = st.columns([1,1])
with w1:
    st.subheader("Top keywords (by label)")
    for label in ["positif","netral","negatif"]:
        text = " ".join(df.loc[df["Sentimen"]==label, "Komentar_Bersih"].astype(str))
        if not text.strip():
            st.markdown(f"**{label.capitalize()}**: -")
            continue
        words = pd.Series(text.split()).value_counts().head(8)
        st.markdown(f"**{label.capitalize()}**: " + ", ".join([f"{w} ({c})" for w,c in words.items()]))
with w2:
    st.subheader("Word Cloud (positif)")
    txt = " ".join(df.loc[df["Sentimen"]=="positif", "Komentar_Bersih"].astype(str))
    if txt.strip():
        wc = WordCloud(width=800, height=300, background_color=None, mode="RGBA").generate(txt)
        fig = plt.figure(figsize=(6,2.4)); plt.imshow(wc); plt.axis("off")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Tidak ada teks positif untuk wordcloud.")

st.markdown("---")

# Table & download
st.subheader("Tabel komentar (hasil analisis)")
cols_show = [c for c in ["No","Tanggal","Komentar","Komentar_Bersih","Sentimen","Kepercayaan"] if c in df.columns]
st.dataframe(df[cols_show].reset_index(drop=True), use_container_width=True, height=360)

# Download button
buf = io.StringIO()
df.to_csv(buf, index=False)
st.download_button("⬇️ Unduh CSV (hasil)", data=buf.getvalue(), file_name="hasil_sentimen.csv", mime="text/csv")

st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
