# dashboard_interaktif.py
# Premium Dark Dashboard (modern "anak IT" look) — Samsat Sentiment
# - Background blur (samsat.jpg in repo root) + blue-purple overlay
# - Sidebar navigation (Overview / Visualisasi / Data)
# - Top hero, large metric cards, Plotly visuals, wordcloud, table, download
# - Real-time Google Sheets or local upload
# - IndoBERT inference optional (Fast Mode default ON)
#
# Run:
#   pip install -r requirements.txt
#   streamlit run dashboard_interaktif.py
#
# Requirements (see bottom of file for suggested requirements.txt)
# --------------------------------------------------------------------

import os
import io
import re
import base64
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Optional heavy libs for model inference
MODEL_AVAILABLE = True
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    MODEL_AVAILABLE = False

# ---------------------------
# CONFIG
# ---------------------------
GSHEET_ID = "1VL8FwJrAAZHqEDErlkhPtQ-a69O4JcRkOBmnYPKH2PY"
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/export?format=csv"

# Page config
st.set_page_config(page_title="Samsat Sentiment — Premium", page_icon="🧠", layout="wide")

# ---------------------------
# ASSETS: background image (samsat.jpg in repo root) - fallback gradient
# ---------------------------
def get_bg_base64():
    if os.path.exists("samsat.jpg"):
        with open("samsat.jpg", "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

BG_B64 = get_bg_base64()

# ---------------------------
# CSS (dark theme, glass cards, typography)
# ---------------------------
CSS = f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
:root {{
  --bg-deep: #0b1220;
  --panel: rgba(255,255,255,0.04);
  --card: rgba(255,255,255,0.06);
  --muted: rgba(255,255,255,0.65);
  --accent1: #5D3FD3;   /* purple */
  --accent2: #2E86C1;   /* blue */
  --accent-spot: #F4D03F; /* yellow */
  --glass-border: rgba(255,255,255,0.06);
}}
html, body, [class*="css"] {{ font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, Arial; background: transparent; }}
/* background image */
.app-bg {{
  position: fixed; inset: 0; z-index: -99;
  background-position: center; background-size: cover; transform: scale(1.02);
  filter: blur(8px) brightness(.48) saturate(.98);
}}
.app-overlay {{
  position: fixed; inset: 0; z-index: -98;
  background: linear-gradient(120deg, rgba(93,63,211,0.30), rgba(46,134,193,0.18));
  mix-blend-mode: multiply;
}}
/* top bar */
.topbar {{
  display:flex; align-items:center; justify-content:space-between;
  gap:12px; padding:18px; margin:18px 28px 6px 28px;
  border-radius:12px; background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border: 1px solid var(--glass-border); box-shadow: 0 12px 30px rgba(6,10,20,0.45);
}}
.brand-title {{ color: rgba(235,245,255,0.97); font-weight:800; font-size:20px; margin:0; }}
.brand-sub {{ color: rgba(235,245,255,0.8); margin:0; font-size:13px; }}

/* sidebar tweaks */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(8,10,20,0.92), rgba(10,14,26,0.95));
  color: #eaf2ff;
}}
.sidebar .nav-item {{ padding:10px 12px; border-radius:8px; margin-bottom:6px; }}
.sidebar .nav-item:hover {{ background: rgba(255,255,255,0.02); transform: translateX(6px); transition: .18s; }}

/* cards */
.card {{
  background: var(--card); border-radius:12px; padding:14px;
  border: 1px solid var(--glass-border);
  box-shadow: 0 10px 28px rgba(6,10,20,0.45);
}}
.metric-title {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }}
.metric-value {{ font-size:26px; font-weight:800; color: #fff; }}

/* small text */
.small-muted {{ color: rgba(235,245,255,0.78); font-size:13px; }}

/* dataframes */
div[data-testid="stDataFrame"] table {{
  background: rgba(255,255,255,0.02);
  color: #eaf2ff;
}}
/* responsiveness */
@media (max-width: 768px) {{
  .topbar {{ margin:10px; padding:12px; flex-direction:column; align-items:flex-start; gap:8px; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# show background
if BG_B64:
    st.markdown(f'<div class="app-bg" style="background-image:url(data:image/jpg;base64,{BG_B64})"></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="app-bg" style="background: linear-gradient(120deg,#081226,#12213a)"></div>', unsafe_allow_html=True)
st.markdown('<div class="app-overlay"></div>', unsafe_allow_html=True)

# ---------------------------
# Utility: text cleaning (simple Indonesian-aware)
# ---------------------------
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    factory = StopWordRemoverFactory()
    STOPWORDS = set(factory.get_stop_words())
except Exception:
    STOPWORDS = set()

def normalize_repeats(s: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1\1', s)

def clean_text(s: str) -> str:
    t = str(s).lower()
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'@[^\s]+|#[^\s]+', ' ', t)
    t = t.encode('ascii', 'ignore').decode('ascii')  # drop emoji for model text
    t = re.sub(r'[^a-z\s]', ' ', t)
    t = normalize_repeats(t)
    toks = [w for w in t.split() if w and w not in STOPWORDS]
    return " ".join(toks).strip()

# ---------------------------
# Model loader (safe): only load if user disables Fast Mode (and model packages available)
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_indobert_safe():
    # model name used previously
    MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mapping = {"LABEL_0": "positif", "LABEL_1": "netral", "LABEL_2": "negatif"}
    return tokenizer, model, mapping

# ---------------------------
# Prediction function cached on CSV bytes + fast flag
# ---------------------------
@st.cache_data(show_spinner=False)
def predict_from_csv_bytes(csv_bytes: bytes, fast_mode: bool, batch_size: int = 32):
    bio = io.BytesIO(csv_bytes)
    df = pd.read_csv(bio)
    df.columns = [c.strip() for c in df.columns]
    # normalize common column names case-insensitively
    colmap = {}
    for want in ["No","Tanggal","Komentar"]:
        for c in df.columns:
            if c.strip().lower() == want.lower():
                colmap[c] = want
    if colmap:
        df = df.rename(columns=colmap)
    if not set(["No","Tanggal","Komentar"]).issubset(set(df.columns)):
        return pd.DataFrame({"error": ["Missing required columns: No, Tanggal, Komentar"]})
    df = df.dropna(subset=["Komentar"]).reset_index(drop=True)
    df["Komentar_Bersih"] = df["Komentar"].astype(str).apply(clean_text)
    if fast_mode or (not MODEL_AVAILABLE):
        # deterministic pseudo-labels for preview if model not loaded
        rng = np.random.default_rng(2025)
        labels = rng.choice(["positif","netral","negatif"], size=len(df))
        confs = (rng.random(len(df)) * 0.35 + 0.6).round(4)
        df["Sentimen"] = labels
        df["Kepercayaan"] = confs
        return df

    # real inference
    tokenizer, model, mapping = load_indobert_safe()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    texts = df["Komentar_Bersih"].astype(str).tolist()
    labels = []
    confs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=160, return_tensors="pt")
        enc = {k: v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            topv, topi = probs.max(dim=1)
        for idx, p in zip(topi.cpu().tolist(), topv.cpu().tolist()):
            labels.append(mapping.get(f"LABEL_{idx}", "netral"))
            confs.append(float(p))
    df["Sentimen"] = labels
    df["Kepercayaan"] = np.round(confs, 4)
    return df

# ---------------------------
# DATA SOURCE controls (in sidebar): minimal navigation and essential settings
# ---------------------------
st.sidebar.markdown("<div style='font-weight:800; font-size:16px; color:#eaf2ff; margin-bottom:6px;'>Sumber Data & Pengaturan</div>", unsafe_allow_html=True)
mode = st.sidebar.selectbox("Mode data", ["Google Sheets (real-time)", "Upload CSV/XLSX"])
uploaded = None
if mode.startswith("Upload"):
    uploaded = st.sidebar.file_uploader("Upload CSV atau XLSX", type=["csv","xlsx"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Analisis**")
fast_mode = st.sidebar.checkbox("Fast mode (Preview tanpa model, RECOMMENDED for demo)", value=True)
sample_limit = st.sidebar.slider("Limit baris yang diproses (demo)", min_value=100, max_value=5000, value=1000, step=100)
st.sidebar.markdown("---")
st.sidebar.markdown("Navigasi")
nav = st.sidebar.radio("", ["Overview", "Visualisasi", "Data"])

# ---------------------------
# Fetch CSV bytes (cached)
# ---------------------------
@st.cache_data(ttl=45)
def fetch_csv_bytes_from_url(url: str) -> bytes:
    df_temp = pd.read_csv(url)
    bio = io.BytesIO()
    df_temp.to_csv(bio, index=False)
    return bio.getvalue()

raw_bytes = None
if mode.startswith("Google"):
    try:
        raw_bytes = fetch_csv_bytes_from_url(GSHEET_CSV_URL)
    except Exception as e:
        st.sidebar.error("Gagal ambil Google Sheets: " + str(e))
        st.stop()
else:
    if uploaded is None:
        st.sidebar.info("Silakan upload file CSV/XLSX atau pilih Google Sheets.")
        st.stop()
    else:
        if uploaded.name.lower().endswith(".xlsx"):
            df_local = pd.read_excel(uploaded)
            buf = io.BytesIO(); df_local.to_csv(buf, index=False); raw_bytes = buf.getvalue()
        else:
            raw_bytes = uploaded.read()

# ---------------------------
# Run inference (cache-aware)
# ---------------------------
with st.spinner("Menyiapkan data & analisis..."):
    df_all = predict_from_csv_bytes(raw_bytes, fast_mode=fast_mode, batch_size=32)

if "error" in df_all.columns:
    st.error(df_all["error"].iat[0])
    st.stop()

# Normalize date column if present
df_all.columns = [c.strip() for c in df_all.columns]
if "Tanggal" in df_all.columns:
    df_all["Tanggal"] = pd.to_datetime(df_all["Tanggal"], errors="coerce").dt.date

# optionally limit rows for performance
df_proc = df_all.head(min(len(df_all), sample_limit)).copy()

# ---------------------------
# Top hero (center) — show title + quick filters
# ---------------------------
st.markdown('<div class="topbar">', unsafe_allow_html=True)
st.markdown('<div><div class="brand-title">Samsat Sentiment — Dashboard</div><div class="brand-sub">Realtime • IndoBERT • Modern</div></div>', unsafe_allow_html=True)

# inline simple filters in header (date range + search)
col1, col2, col3 = st.columns([1.6,1,0.8])
with col1:
    if "Tanggal" in df_proc.columns:
        dmin = df_proc["Tanggal"].min()
        dmax = df_proc["Tanggal"].max()
        daterange = st.date_input("Rentang Tanggal (filter)", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    else:
        daterange = None
with col2:
    q = st.text_input("Cari kata kunci (komentar)", value="")
with col3:
    st.markdown("", unsafe_allow_html=True)  # placeholder for alignment

st.markdown('</div>', unsafe_allow_html=True)

# apply filters
df_view = df_proc.copy()
if daterange and isinstance(daterange, tuple) and len(daterange) == 2 and "Tanggal" in df_view.columns:
    s,e = daterange
    df_view = df_view[(df_view["Tanggal"] >= s) & (df_view["Tanggal"] <= e)]
if q:
    df_view = df_view[df_view["Komentar"].str.contains(q, case=False, na=False)]

if df_view.empty:
    st.warning("Tidak ada data setelah filter. Silakan ubah filter.")
    st.stop()

# ---------------------------
# Metrics row
# ---------------------------
total = len(df_view)
cnt_pos = int((df_view["Sentimen"] == "positif").sum())
cnt_neu = int((df_view["Sentimen"] == "netral").sum())
cnt_neg = int((df_view["Sentimen"] == "negatif").sum())
avg_conf = float(df_view["Kepercayaan"].mean()) if "Kepercayaan" in df_view.columns else 0.0

m1, m2, m3, m4 = st.columns([1,1,1,1], gap="small")
with m1:
    st.markdown(f'<div class="card"><div class="metric-title">Total Komentar</div><div class="metric-value">{total:,}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="card"><div class="metric-title">Positif</div><div class="metric-value" style="color:#7AE582">{cnt_pos:,}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="card"><div class="metric-title">Netral</div><div class="metric-value" style="color:#FFD97A">{cnt_neu:,}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="card"><div class="metric-title">Negatif</div><div class="metric-value" style="color:#FF8A8A">{cnt_neg:,}</div></div>', unsafe_allow_html=True)

st.markdown(f'<div style="margin-top:8px;"><span class="small-muted">Insight: Dominan <b>{"Positif" if cnt_pos>=max(cnt_neu,cnt_neg) else ("Netral" if cnt_neu>=cnt_neg else "Negatif")}</b> • Avg confidence: <b>{avg_conf*100:.1f}%</b></span></div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Page content based on selected nav
# ---------------------------
if nav == "Overview":
    st.subheader("Overview — Ringkasan & Contoh Komentar")
    # show pie + small example comments
    col_a, col_b = st.columns([1,1])
    with col_a:
        dist = df_view["Sentimen"].value_counts().reset_index()
        dist.columns = ["Sentimen","Jumlah"]
        fig = px.pie(dist, names="Sentimen", values="Jumlah", hole=0.35,
                     color_discrete_map={"positif":"#7AE582","netral":"#FFD97A","negatif":"#FF8A8A"})
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.markdown("**Contoh komentar representatif**")
        for lbl in ["positif","netral","negatif"]:
            sub = df_view[df_view["Sentimen"]==lbl]
            sample = sub["Komentar"].sample(n=1, random_state=42).iat[0] if not sub.empty else "-"
            st.markdown(f"- **{lbl.capitalize()}**: {sample}")

    st.markdown("---")
    st.subheader("Top Words (per label)")
    cols = st.columns(3)
    for i, lbl in enumerate(["positif","netral","negatif"]):
        txt = " ".join(df_view.loc[df_view["Sentimen"]==lbl, "Komentar_Bersih"].astype(str))
        if txt.strip():
            topw = pd.Series(txt.split()).value_counts().head(8)
            cols[i].markdown("**" + lbl.capitalize() + "**")
            cols[i].write(", ".join([f"{w} ({c})" for w,c in topw.items()]))
        else:
            cols[i].info("No data")

elif nav == "Visualisasi":
    st.subheader("Visualisasi — Trend & Scatter")
    c1, c2 = st.columns([1.2,1])
    with c1:
        if "Tanggal" in df_view.columns:
            tmp = df_view.groupby([df_view["Tanggal"].astype(str), "Sentimen"]).size().reset_index(name="Jumlah")
            tmp.columns = ["Tanggal","Sentimen","Jumlah"]
            fig = px.line(tmp, x="Tanggal", y="Jumlah", color="Sentimen", markers=True)
            fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Kolom 'Tanggal' tidak tersedia; trend tidak bisa ditampilkan.")
    with c2:
        df_view["Panjang"] = df_view["Komentar_Bersih"].str.split().map(len)
        trendline = "ols" if (("Kepercayaan" in df_view.columns) and (len(df_view)>=10)) else None
        fig2 = px.scatter(df_view, x="Panjang", y="Kepercayaan" if "Kepercayaan" in df_view.columns else None,
                          color="Sentimen", hover_data=["No","Tanggal","Komentar"], trendline=trendline)
        fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("WordCloud (positif)")
    txt_pos = " ".join(df_view.loc[df_view["Sentimen"]=="positif", "Komentar_Bersih"].astype(str))
    if txt_pos.strip():
        wc = WordCloud(width=900, height=300, background_color=None, mode="RGBA").generate(txt_pos)
        fig_wc = plt.figure(figsize=(10,3)); plt.imshow(wc); plt.axis("off")
        st.pyplot(fig_wc, use_container_width=True)
    else:
        st.info("Tidak cukup data positif untuk wordcloud.")

else:  # Data
    st.subheader("Data — Tabel & Download")
    show_cols = [c for c in ["No","Tanggal","Komentar","Komentar_Bersih","Sentimen","Kepercayaan"] if c in df_view.columns]
    st.dataframe(df_view[show_cols].reset_index(drop=True), use_container_width=True, height=520)
    # download
    csv_buf = io.StringIO()
    df_view.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download hasil (CSV)", csv_buf.getvalue(), "hasil_sentimen.csv", "text/csv")

# Footer small
st.markdown("<hr style='opacity:.2'/>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:rgba(255,255,255,0.6); padding-bottom:18px;'>Built with ❤️ • IndoBERT (optional) • Realtime Google Sheets</div>", unsafe_allow_html=True)
