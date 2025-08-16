# dashboard_interaktif.py
# =============================================================
# Premium Dashboard Sentimen — Real-time Google Sheets + IndoBERT
# Features: modern UI, dark/light toggle, animations, caching per-sheet,
#           insight automatic, responsive layout, wordclouds, download.
#
# Expected Google Sheet columns (case-insensitive):
#   No | Tanggal | Komentar
#
# Usage:
#   pip install -r requirements.txt
#   streamlit run dashboard_interaktif.py
# =============================================================

import io
import re
import time
import hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import streamlit as st

# CSS untuk menambahkan background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("samsat.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);  /* header transparan */
}
[data-testid="stSidebar"] {
    background-color: rgba(15, 20, 35, 0.95); /* sidebar semi-transparan */
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("📊 Dashboard Modern dengan Background")
st.write("Tampilan dengan tema biru keunguan + background image.")

# Optional: nicer sidebar menu
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# Transformers
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional: statsmodels for trendline (if installed)
try:
    import statsmodels.api  # noqa: F401
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# ----------------------------
# CONFIG: Google Sheet ID (already provided)
# ----------------------------
GSHEET_ID = "1VL8FwJrAAZHqEDErlkhPtQ-a69O4JcRkOBmnYPKH2PY"
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/export?format=csv"

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Dashboard Sentimen — Real-time (IndoBERT)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# THEME & CUSTOM CSS (light + dark toggle)
# ----------------------------
PRIMARY = "#2E86C1"
ACCENT = "#F4D03F"
TEXT = "#1B2631"
BG_LIGHT = "#FBFCFD"
CARD_BG = "#FFFFFF"

# We'll inject CSS that supports a dark mode class toggle
st.markdown(
    f"""
    <style>
    :root {{
      --primary: {PRIMARY};
      --accent: {ACCENT};
      --text: {TEXT};
      --bg-light: {BG_LIGHT};
      --card-bg: {CARD_BG};
    }}

    /* Basic hero */
    .hero {{
      background: linear-gradient(90deg, rgba(46,134,193,0.06), rgba(244,208,63,0.03));
      padding: 18px;
      border-radius: 12px;
      margin-bottom: 14px;
      box-shadow: 0 8px 22px rgba(11,22,39,0.04);
    }}
    .hero h1 {{ color: var(--primary); margin:0; font-weight:800; font-size:34px; }}
    .hero p {{ margin:4px 0 0; color:#4b5563; }}

    /* Metric cards */
    .metric-card {{
      background: var(--card-bg);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 8px 20px rgba(11,22,39,0.04);
      transition: transform .18s ease, box-shadow .18s ease;
    }}
    .metric-card:hover {{ transform: translateY(-6px); box-shadow: 0 18px 40px rgba(11,22,39,0.08); }}
    .metric-title {{ font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.08em; }}
    .metric-value {{ font-size:22px; font-weight:800; color:var(--text); }}

    /* nice button */
    .stDownloadButton>button, .stButton>button {{
      border-radius:10px !important; padding:10px 14px !important;
      background: var(--primary) !important; color: #fff !important; border: none !important;
    }}

    /* Page layout tweaks */
    div[data-testid="stSidebar"] {{ background-color: rgba(246,248,250,0.9); }}
    .sidebar .css-1d391kg {{}}

    /* Animation */
    @keyframes fadeInUp {{
      0% {{ opacity: 0; transform: translateY(8px); }}
      100% {{ opacity: 1; transform: translateY(0); }}
    }}
    .fade-up {{ animation: fadeInUp .36s ease both; }}

    /* Dark mode class */
    .dark-mode {{
      --bg-light: #0b1220;
      --card-bg: #071122;
      --text: #e6eef8;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# UTIL: text preprocessing (simple but effective for ID)
# ----------------------------
# Stopwords from Sastrawi if available
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    factory = StopWordRemoverFactory()
    STOPWORDS = set(factory.get_stop_words())
except Exception:
    STOPWORDS = set()

def normalize_repeated_chars(s: str) -> str:
    # compress long repeated letters e.g. "soooo" -> "soo"
    return re.sub(r"(.)\1{2,}", r"\1\1", s)

def clean_text(text: str) -> str:
    t = str(text).lower()
    # remove urls
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    # remove mentions and hashtags
    t = re.sub(r"@[^\s]+|#[^\s]+", " ", t)
    # remove non-ascii (attempt to strip emojis)
    t = t.encode("ascii", "ignore").decode("ascii")
    # only keep letters and spaces
    t = re.sub(r"[^a-z\s]", " ", t)
    t = normalize_repeated_chars(t)
    tokens = [w for w in t.split() if w and w not in STOPWORDS]
    return " ".join(tokens).strip()

# ----------------------------
# MODEL: load IndoBERT (cached)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_indobert():
    PRETRAIN = "mdhugol/indonesia-bert-sentiment-classification"
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAIN)
    # mapping according to model card
    mapping = {"LABEL_0": "positif", "LABEL_1": "netral", "LABEL_2": "negatif"}
    return tokenizer, model, mapping

# ----------------------------
# HELPERS: cache predictions by data hash
# ----------------------------
def df_hash_bytes(df: pd.DataFrame) -> str:
    """Return SHA256 hex digest of dataframe CSV bytes (stable)."""
    bio = io.BytesIO()
    # use consistent csv formatting
    df.to_csv(bio, index=False)
    return hashlib.sha256(bio.getvalue()).hexdigest()

@st.cache_data(show_spinner=False)
def predict_cached(data_csv_bytes: bytes, fast_mode: bool, bsize: int = 32) -> pd.DataFrame:
    """
    Predict function cached on CSV bytes and fast_mode flag.
    Receives raw CSV bytes (snapshot) to make caching robust.
    """
    # decode to DataFrame
    bio = io.BytesIO(data_csv_bytes)
    df = pd.read_csv(bio)
    # ensure columns normalized to expected names (case-insensitive)
    df.columns = [c.strip() for c in df.columns]
    # map expected column names if different case
    cols_map = {}
    for expected in ["No", "Tanggal", "Komentar"]:
        for c in df.columns:
            if c.strip().lower() == expected.lower():
                cols_map[c] = expected
    if cols_map:
        df = df.rename(columns=cols_map)

    # ensure expected present
    if not set(["No", "Tanggal", "Komentar"]).issubset(set(df.columns)):
        # return empty dataframe with message columns to avoid crash
        return pd.DataFrame({
            "error": ["Missing required columns (No, Tanggal, Komentar)"]
        })

    # preprocess
    df = df.copy()
    df["Komentar"] = df["Komentar"].astype(str)
    df["Komentar_Bersih"] = df["Komentar"].apply(clean_text)

    if fast_mode:
        # dummy random labels for fast preview
        rng = np.random.default_rng(1234)
        labels = rng.choice(["positif", "netral", "negatif"], size=len(df))
        confs = rng.random(len(df)) * 0.5 + 0.5
        df["Sentimen"] = labels
        df["Kepercayaan"] = np.round(confs, 4)
        return df

    # real model inference
    tokenizer, model, mapping = load_indobert()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    texts = df["Komentar_Bersih"].astype(str).tolist()
    labels_list = []
    confs_list = []
    batch_size = bsize
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=160, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc).logits
            probs = torch.softmax(out, dim=-1)
            topv, topi = probs.max(dim=1)
        for idx, p in zip(topi.cpu().tolist(), topv.cpu().tolist()):
            labels_list.append(mapping.get(f"LABEL_{idx}", "netral"))
            confs_list.append(float(p))
    df["Sentimen"] = labels_list
    df["Kepercayaan"] = np.round(confs_list, 4)
    return df

# ----------------------------
# Load data (Google Sheet or local)
# ----------------------------
with st.sidebar:
    st.title("📂 Sumber Data & Pengaturan")
    st.caption("Dashboard ini mengambil data real-time dari Google Sheets (link yang kamu berikan).")
    st.write(f"Spreadsheet ID: `{GSHEET_ID}`")
    mode = st.selectbox("Mode data", ["Google Sheets (real-time)", "Local file (CSV/XLSX)"])
    uploaded = None
    if mode == "Local file (CSV/XLSX)":
        uploaded = st.file_uploader("Upload file lokal", type=["csv", "xlsx"])
    st.markdown("---")
    st.subheader("⚙️ Pengaturan Analisis")
    fast_mode = st.toggle("Fast mode (no model) — cepat (untuk demo)", value=False)
    sample_limit = st.slider("Batas baris yang diproses (untuk demo)", min_value=50, max_value=5000, value=1000, step=50)
    st.markdown("---")
    st.subheader("🖌 Tampilan")
    dark_mode = st.toggle("Dark mode", value=False)
    st.caption("Toggling dark will apply a darker theme for presentations in low light.")
    st.markdown("---")
    st.caption("Tip: gunakan Fast mode saat presentasi jika koneksi lambat atau ingin demo cepat.")

# apply dark-mode toggling by injecting class on body (simple approach)
if dark_mode:
    st.markdown("<script>document.querySelector('body').classList.add('dark-mode')</script>", unsafe_allow_html=True)
else:
    st.markdown("<script>document.querySelector('body').classList.remove('dark-mode')</script>", unsafe_allow_html=True)

# read sheet
@st.cache_data(ttl=45)
def fetch_sheet_csv(url: str) -> bytes:
    # return raw bytes of CSV (used for hashing and cache)
    try:
        df_temp = pd.read_csv(url)
        bio = io.BytesIO()
        df_temp.to_csv(bio, index=False)
        return bio.getvalue()
    except Exception as e:
        raise

raw_bytes = None
if mode == "Google Sheets (real-time)":
    try:
        raw_bytes = fetch_sheet_csv(GSHEET_CSV_URL)
    except Exception as e:
        st.error("Gagal membaca Google Sheets: " + str(e))
        st.info("Silakan pilih Local file sebagai fallback.")
        if uploaded is None:
            st.stop()
else:
    if uploaded is None:
        st.info("Silakan upload file lokal, atau pilih Google Sheets mode.")
        st.stop()
    else:
        # load uploaded bytes
        raw_bytes = uploaded.read()

# now we have raw_bytes -> run cached prediction
with st.spinner("🔄 Menyiapkan data & memproses (cache-aware)..."):
    df_result = predict_cached(raw_bytes, fast_mode=fast_mode, bsize=32)

# check error case
if "error" in df_result.columns:
    st.error(df_result["error"].iat[0])
    st.stop()

# basic cleaning for date & columns
df_result.columns = [c.strip() for c in df_result.columns]
# map date column
if "Tanggal" in df_result.columns:
    df_result["Tanggal"] = pd.to_datetime(df_result["Tanggal"], errors="coerce").dt.date
# limit rows displayed/processed
if sample_limit and len(df_result) > sample_limit:
    df_proc = df_result.head(sample_limit).copy().reset_index(drop=True)
else:
    df_proc = df_result.copy()

# ----------------------------
# Header & hero
# ----------------------------
st.markdown(
    f"""
    <div class="hero fade-up">
      <h1>🧠 Dashboard Sentimen — Real-time (Google Sheets)</h1>
      <p>Preprocessing otomatis • IndoBERT sentiment (positif / netral / negatif) • Visual & insight • Download hasil.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Filter panel (global)
# ----------------------------
with st.expander("🔧 Filter Data (global)", expanded=True):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        dmin = pd.to_datetime(df_proc["Tanggal"]).min().date() if "Tanggal" in df_proc.columns else None
        dmax = pd.to_datetime(df_proc["Tanggal"]).max().date() if "Tanggal" in df_proc.columns else None
        if dmin and dmax:
            date_range = st.date_input("Rentang tanggal", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        else:
            date_range = None
    with col2:
        # dynamic column selector for category-like cols
        candidate_cols = [c for c in df_proc.columns if df_proc[c].dtype == object and c.lower() not in ("komentar",)]
        cat_col = st.selectbox("Filter kolom kategori (opsional)", options=["(none)"] + candidate_cols)
        cat_filter_vals = None
        if cat_col and cat_col != "(none)":
            opts = sorted(df_proc[cat_col].dropna().unique().tolist())
            cat_filter_vals = st.multiselect(f"Pilih {cat_col}", options=opts, default=opts[:5] if len(opts)>5 else opts)
    with col3:
        st.write("Lainnya")
        apply_button = st.button("Apply / Refresh")

# apply filters to df_proc -> df_view
df_view = df_proc.copy()
if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    s, e = date_range
    ser = pd.to_datetime(df_view["Tanggal"]).dt.date
    df_view = df_view[(ser >= s) & (ser <= e)]
if cat_filter_vals:
    df_view = df_view[df_view[cat_col].isin(cat_filter_vals)]

if df_view.empty:
    st.warning("Tidak ada data setelah filter. Coba rentang tanggal lain atau hapus filter kategori.")
    st.stop()

# ----------------------------
# Metrics cards (interactive)
# ----------------------------
total = len(df_view)
pos = int((df_view["Sentimen"] == "positif").sum())
neu = int((df_view["Sentimen"] == "netral").sum())
neg = int((df_view["Sentimen"] == "negatif").sum())
avg_conf = float(df_view["Kepercayaan"].mean()) if "Kepercayaan" in df_view.columns else np.nan

m1, m2, m3, m4 = st.columns([1,1,1,1])
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Total (terfilter)</div><div class="metric-value">{total:,}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Positif</div><div class="metric-value">{pos:,}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Netral</div><div class="metric-value">{neu:,}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Negatif</div><div class="metric-value">{neg:,}</div></div>', unsafe_allow_html=True)

# Insight box (more descriptive)
dominant = "Positif" if pos >= max(neu, neg) else ("Netral" if neu >= neg else "Negatif")
st.markdown(f"**Insight cepat:** Mayoritas komentar bersifat **{dominant}** pada rentang tanggal terpilih. Rata-rata kepercayaan model: **{avg_conf*100:.1f}%**.")

# ----------------------------
# Page navigation (Overview / Analisis / Visual)
# ----------------------------
if HAS_OPTION_MENU:
    page = option_menu(None, ["Overview", "Analisis", "Visualisasi"], icons=["house", "chat-left-dots", "bar-chart"], default_index=0, orientation="horizontal")
else:
    page = st.radio("Halaman", ["Overview", "Analisis", "Visualisasi"], horizontal=True)

# ----------------------------
# Page: Overview
# ----------------------------
def page_overview(dfv: pd.DataFrame):
    st.subheader("Overview — Preview & Download")
    left, right = st.columns([2,1])
    with left:
        st.write("5 baris pertama (setelah preprocessing & inferensi):")
        cols_show = [c for c in ["No", "Tanggal", "Komentar", "Komentar_Bersih", "Sentimen", "Kepercayaan"] if c in dfv.columns]
        st.dataframe(dfv[cols_show].head(8), use_container_width=True)
    with right:
        st.write("Ringkasan statistik singkat")
        stats = pd.DataFrame({
            "Panjang (kata)": dfv["Komentar_Bersih"].str.split().map(len),
            "Kepercayaan": dfv.get("Kepercayaan", pd.Series([np.nan]*len(dfv)))
        }).describe().T
        st.dataframe(stats, use_container_width=True, height=260)

    # download button
    buff = io.StringIO()
    dfv.to_csv(buff, index=False)
    st.download_button("⬇️ Download CSV Hasil Analisis", data=buff.getvalue(), file_name="hasil_analisis_sentimen.csv", mime="text/csv")

# ----------------------------
# Page: Analisis (distribusi + wordcloud)
# ----------------------------
def page_analisis(dfv: pd.DataFrame):
    st.subheader("Analisis — Distribusi & WordCloud")
    dist = dfv["Sentimen"].value_counts().reset_index()
    dist.columns = ["Sentimen", "Jumlah"]

    # Pie + bar side-by-side
    c1, c2 = st.columns([1,1])
    with c1:
        fig_pie = px.pie(dist, names="Sentimen", values="Jumlah", hole=0.36,
                         color_discrete_map={"positif":"#2ECC71","netral":ACCENT,"negatif":"#E74C3C"})
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_bar = px.bar(dist.sort_values("Jumlah", ascending=False), x="Sentimen", y="Jumlah", text_auto=True,
                         color="Sentimen", color_discrete_map={"positif":"#2ECC71","netral":ACCENT,"negatif":"#E74C3C"})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### ☁️ Word Cloud per Sentimen")
    c1, c2, c3 = st.columns(3)
    for label, col in zip(["positif","netral","negatif"], [c1, c2, c3]):
        text = " ".join(dfv.loc[dfv["Sentimen"]==label, "Komentar_Bersih"].astype(str))
        if not text.strip():
            col.info(f"Tidak ada teks untuk **{label}**.")
            continue
        wc = WordCloud(width=900, height=440, background_color="white", max_words=80).generate(text)
        fig = plt.figure(figsize=(6,3.2))
        plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
        col.pyplot(fig, use_container_width=True)

# ----------------------------
# Page: Visualisasi
# ----------------------------
def page_visual(dfv: pd.DataFrame):
    st.subheader("Visualisasi — Trend & Confidence")
    tmp = dfv.copy()
    if "Tanggal" in tmp.columns:
        tmp["Tanggal_dt"] = pd.to_datetime(tmp["Tanggal"])
        trend = tmp.groupby([tmp["Tanggal_dt"].dt.to_period("D").astype(str), "Sentimen"]).size().reset_index(name="Jumlah")
        trend.columns = ["Tanggal", "Sentimen", "Jumlah"]
        fig_line = px.line(trend, x="Tanggal", y="Jumlah", color="Sentimen", markers=True,
                           color_discrete_map={"positif":"#2ECC71","netral":ACCENT,"negatif":"#E74C3C"})
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Kolom Tanggal tidak ditemukan; trend per-hari tidak tersedia.")

    # scatter: length vs confidence
    tmp["Panjang"] = tmp["Komentar_Bersih"].str.split().map(len)
    trendline_opt = "ols" if HAS_STATSMODELS and len(tmp) >= 10 else None
    fig_sc = px.scatter(tmp, x="Panjang", y="Kepercayaan", color="Sentimen", hover_data=["No","Tanggal","Komentar"],
                        trendline=trendline_opt, color_discrete_map={"positif":"#2ECC71","netral":ACCENT,"negatif":"#E74C3C"})
    st.plotly_chart(fig_sc, use_container_width=True)

    with st.expander("📋 Tabel hasil (filterable)"):
        options = st.multiselect("Filter label", options=["positif","netral","negatif"], default=["positif","netral","negatif"])
        tab = tmp[tmp["Sentimen"].isin(options)]
        st.dataframe(tab[["No","Tanggal","Komentar","Sentimen","Kepercayaan"]], use_container_width=True, height=320)
        buf = io.StringIO()
        tab.to_csv(buf, index=False)
        st.download_button("💾 Download CSV (ter-filter)", data=buf.getvalue(), file_name="hasil_filter_sentimen.csv", mime="text/csv")

# ----------------------------
# Route to selected page
# ----------------------------
if page == "Overview":
    page_overview(df_view)
elif page == "Analisis":
    page_analisis(df_view)
else:
    page_visual(df_view)

# Footer
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:12px; opacity:0.7;padding:10px'>Created with ❤️ — Premium UI • IndoBERT • Real-time Google Sheets</div>", unsafe_allow_html=True)
