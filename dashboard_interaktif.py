# dashboard_interaktif.py
# =========================================================
# Dashboard Analisis Sentimen (Google Sheets real-time + IndoBERT)
# Designed: modern, responsive, out-of-the-box for dataset with columns:
#   No | Tanggal | Komentar
#
# Usage:
#   pip install -r requirements.txt
#   streamlit run dashboard_interaktif.py
# =========================================================

import io
import re
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Optional UI nicety
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# Transformers & torch (lazy import inside load_model)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------
# CONFIG: gsheet ID yang kamu kirim
# ------------------------
GSHEET_ID = "1VL8FwJrAAZHqEDErlkhPtQ-a69O4JcRkOBmnYPKH2PY"
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/export?format=csv"

# ------------------------
# Page config & theme
# ------------------------
st.set_page_config(page_title="Dashboard Sentimen (IndoBERT) — Real-time",
                   page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

PRIMARY = "#2E86C1"
ACCENT  = "#F4D03F"
TEXT    = "#1B2631"
BG      = "#FFFFFF"

# custom CSS for nicer look
st.markdown(f"""
    <style>
      :root{{--primary:{PRIMARY};--accent:{ACCENT};--text:{TEXT};--bg:{BG};}}
      .hero{{background: linear-gradient(90deg, rgba(46,134,193,0.06), rgba(244,208,63,0.04)); padding:18px; border-radius:12px; margin-bottom:12px;}}
      .hero h1{{color:var(--primary); margin:0; font-weight:800;}}
      .metric-card{{background:#fff; border-radius:12px; padding:12px; box-shadow:0 8px 20px rgba(0,0,0,0.05);}}
      .metric-title{{font-size:12px; color:#6c7a89; text-transform:uppercase; letter-spacing:.06em;}}
      .metric-value{{font-size:22px; font-weight:800; color:var(--text);}}
      .stDownloadButton button{{background:var(--primary) !important; color:white !important; border-radius:10px !important;}}
      @media (max-width:600px) {{ .metric-value{{font-size:18px;}} }}
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Helper: text cleaning (Bahasa Indonesia)
# ------------------------
# simple stopword set using Sastrawi if available; fallback empty set
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    s_factory = StopWordRemoverFactory()
    STOPWORDS = set(s_factory.get_stop_words())
except Exception:
    STOPWORDS = set()

def normalize_repeat_chars(s: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1\1', s)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)        # remove urls
    text = re.sub(r'@\w+|#\w+', ' ', text)                   # mentions/hashtags
    # remove emojis/non-ascii
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^a-z\s]', ' ', text)                    # letters only
    text = normalize_repeat_chars(text)
    tokens = [w for w in text.split() if w and w not in STOPWORDS]
    return " ".join(tokens).strip()

# ------------------------
# Load model (cached). Uses mdhugol/indonesia-bert-sentiment-classification
# Label mapping: LABEL_0 -> positif, LABEL_1 -> netral, LABEL_2 -> negatif
# ------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    PRETRAIN = "mdhugol/indonesia-bert-sentiment-classification"
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAIN)
    label_map = {"LABEL_0": "positif", "LABEL_1": "netral", "LABEL_2": "negatif"}
    return tokenizer, model, label_map

# ------------------------
# Load data from Google Sheets (real-time) with fallback to example
# ------------------------
@st.cache_data(ttl=60)  # cache 60s so dashboard reads fresh every minute
def load_gsheet_csv(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        raise

def example_df():
    # small example when sheet unavailable
    data = {
        "No": [1,2,3,4,5],
        "Tanggal": pd.date_range("2025-07-01", periods=5).date,
        "Komentar": [
            "Pelayanan sangat cepat dan ramah",
            "Aplikasi sering error ketika ingin bayar",
            "Biasa saja, tidak istimewa",
            "Petugas sangat membantu",
            "Proses verifikasi terlalu lama"
        ]
    }
    return pd.DataFrame(data)

# ------------------------
# SIDEBAR: controls
# ------------------------
with st.sidebar:
    st.markdown("### 📂 Sumber Data")
    st.write("Data diambil langsung dari Google Sheets (real-time).")
    st.markdown(f"- Spreadsheet ID: `{GSHEET_ID}`")
    st.caption("Jika kamu ingin pakai file lokal, pilih mode 'Local file' di bawah.")

    mode = st.selectbox("Mode koneksi", ["Google Sheets (real-time)", "Local file (CSV/XLSX)"])
    local_file = None
    if mode == "Local file (CSV/XLSX)":
        local_file = st.file_uploader("Upload file lokal (CSV/XLSX)", type=["csv","xlsx"])

    st.markdown("---")
    st.markdown("### ⚙️ Pengaturan Analisis")
    fast_mode = st.toggle("Fast mode (no model) — cepat tanpa mengunduh model", value=False)
    sample_limit = st.slider("Batas jumlah baris yang diproses (untuk demo)", min_value=50, max_value=5000, value=1000, step=50)
    st.markdown("---")
    st.markdown("### ℹ️ Info & Tips")
    st.caption("Fast mode = hanya run minimal clean + random label untuk preview (cepat).")
    st.caption("Non-fast mode akan mengunduh model dari Hugging Face pada first run (butuh Internet).")

# ------------------------
# Read data according to mode
# ------------------------
if mode == "Google Sheets (real-time)":
    try:
        raw_df = load_gsheet_csv(GSHEET_CSV_URL)
    except Exception as e:
        st.warning("Gagal membaca Google Sheets. Menampilkan contoh data. Error: " + str(e))
        raw_df = example_df()
else:
    if local_file:
        try:
            if local_file.name.lower().endswith(".xlsx"):
                raw_df = pd.read_excel(local_file)
            else:
                raw_df = pd.read_csv(local_file)
        except Exception:
            st.error("Gagal membaca file lokal. Pastikan format CSV/XLSX benar.")
            st.stop()
    else:
        st.info("Belum mengunggah file lokal — menampilkan contoh data.")
        raw_df = example_df()

# ------------------------
# Validation & normalize columns
# ------------------------
# Normalize column names (strip spaces)
raw_df.columns = [c.strip() for c in raw_df.columns]

expected_cols = {"No", "Tanggal", "Komentar"}
if not expected_cols.issubset(set(raw_df.columns)):
    # try case-insensitive mapping
    cols_lower = {c.lower(): c for c in raw_df.columns}
    mapping = {}
    for exp in expected_cols:
        if exp.lower() in cols_lower:
            mapping[cols_lower[exp.lower()]] = exp
    if mapping:
        raw_df = raw_df.rename(columns=mapping)

missing = expected_cols - set(raw_df.columns)
if missing:
    st.error(f"Kolom yang dibutuhkan tidak lengkap: {', '.join(missing)}. Pastikan sheet punya kolom: No, Tanggal, Komentar")
    st.stop()

# ensure Tanggal parsed
raw_df["Tanggal"] = pd.to_datetime(raw_df["Tanggal"], errors="coerce").dt.date
raw_df = raw_df.dropna(subset=["Komentar"]).reset_index(drop=True)

# ------------------------
# Header
# ------------------------
st.markdown("""
<div class="hero">
  <h1>🧠 Dashboard Sentimen — Real-time (Google Sheets)</h1>
  <p>Automated preprocessing • IndoBERT sentiment (positif/netral/negatif) • Visual & insight • Download hasil.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------
# Filter panel (global)
# ------------------------
with st.expander("🔧 Filter Data (global)", expanded=True):
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        dmin = pd.to_datetime(raw_df["Tanggal"]).min().date()
        dmax = pd.to_datetime(raw_df["Tanggal"]).max().date()
        date_range = st.date_input("Rentang tanggal", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    with col2:
        region_filter = None
        # optional: if sheet contains region-like column, show dynamic filter
        candidate_region_cols = [c for c in raw_df.columns if raw_df[c].nunique() < 100 and raw_df[c].dtype == object and c.lower() not in ("komentar","tanggal")]
        if candidate_region_cols:
            region_col = st.selectbox("Filter kolom kategori (opsional)", ["(none)"] + candidate_region_cols)
            if region_col != "(none)":
                opts = sorted(raw_df[region_col].dropna().unique().tolist())
                region_filter = st.multiselect(f"Pilih {region_col}", options=opts, default=opts)
    with col3:
        st.write("Pengaturan lainnya")
        apply_button = st.button("Apply filter / Refresh")

# apply filters
df_work = raw_df.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    ser = pd.to_datetime(df_work["Tanggal"]).dt.date
    df_work = df_work[(ser >= start_d) & (ser <= end_d)]
if region_filter is not None:
    df_work = df_work[df_work[region_col].isin(region_filter)]

if len(df_work) == 0:
    st.info("Tidak ada data setelah filter. Cek rentang tanggal atau filter kategori.")
    st.stop()

# limit rows processed (for speed)
limit_n = min(len(df_work), sample_limit)
df_proc = df_work.head(limit_n).copy().reset_index(drop=True)

# ------------------------
# Preprocess texts
# ------------------------
with st.spinner("🔎 Membersihkan teks..."):
    df_proc["Komentar_Bersih"] = df_proc["Komentar"].astype(str).apply(clean_text)

# ------------------------
# Model inference (or fast-mode dummy)
# ------------------------
if fast_mode:
    st.info("Fast mode aktif — model tidak dijalankan. Label sementara dibuat acak (demo).")
    rng = np.random.default_rng(42)
    choices = ["positif","netral","negatif"]
    df_proc["Sentimen"] = rng.choice(choices, size=len(df_proc))
    df_proc["Kepercayaan"] = rng.random(len(df_proc))*0.5 + 0.5
else:
    # load model (may download on first run)
    with st.spinner("📥 Memuat model IndoBERT (first run akan mengunduh model)..."):
        try:
            tokenizer, model, LABEL_MAP = load_model()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
        except Exception as e:
            st.error("Gagal memuat model IndoBERT: " + str(e))
            st.stop()

    # batch prediction helper
    def predict_batches(texts: List[str], bsize: int = 32) -> Tuple[List[str], List[float]]:
        labs, confs = [], []
        for i in range(0, len(texts), bsize):
            batch = texts[i:i+bsize]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=160, return_tensors="pt")
            enc = {k:v.to(device) for k,v in enc.items()}
            with torch.no_grad():
                out = model(**enc).logits
                probs = torch.softmax(out, dim=-1)
                topv, topi = probs.max(dim=1)
            for idx, p in zip(topi.cpu().tolist(), topv.cpu().tolist()):
                label_key = f"LABEL_{idx}"
                labs.append(LABEL_MAP.get(label_key, "netral"))
                confs.append(float(p))
        return labs, confs

    with st.spinner("🤖 Melakukan inferensi (IndoBERT)..."):
        texts = df_proc["Komentar_Bersih"].astype(str).tolist()
        labels, confidences = predict_batches(texts, bsize=32)
        df_proc["Sentimen"] = labels
        df_proc["Kepercayaan"] = np.round(confidences, 4)

# ------------------------
# Kartu metrik
# ------------------------
total = len(df_proc)
cnt_pos = (df_proc["Sentimen"] == "positif").sum()
cnt_neu = (df_proc["Sentimen"] == "netral").sum()
cnt_neg = (df_proc["Sentimen"] == "negatif").sum()
avg_conf = float(df_proc["Kepercayaan"].mean()) if "Kepercayaan" in df_proc.columns else np.nan

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown('<div class="metric-card"><div class="metric-title">Total (diproses)</div><div class="metric-value">{:,}</div></div>'.format(total), unsafe_allow_html=True)
with col_b:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Positif</div><div class="metric-value">{cnt_pos:,}</div></div>', unsafe_allow_html=True)
with col_c:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Netral</div><div class="metric-value">{cnt_neu:,}</div></div>', unsafe_allow_html=True)
with col_d:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Negatif</div><div class="metric-value">{cnt_neg:,}</div></div>', unsafe_allow_html=True)

# insight
dominant = "positif" if cnt_pos >= max(cnt_neu, cnt_neg) else ("netral" if cnt_neu >= cnt_neg else "negatif")
st.markdown(f"**Insight singkat:** Mayoritas komentar bersifat **{dominant}** pada rentang tanggal terpilih. Rata-rata kepercayaan model: **{avg_conf*100:.1f}%**.")

# ------------------------
# PAGE NAVIGATION (Overview / Analysis / Visual)
# ------------------------
if HAS_OPTION_MENU:
    page = option_menu(None, ["Overview","Analisis","Visualisasi"], icons=["house","chat-left-dots","bar-chart"], menu_icon="cast", default_index=0)
else:
    page = st.radio("Halaman", ["Overview","Analisis","Visualisasi"], horizontal=True)

def page_overview(df_display):
    st.subheader("Overview — Preview & Download")
    left, right = st.columns([2,1])
    with left:
        st.write("5 baris pertama (setelah preprocessing & inferensi):")
        st.dataframe(df_display[["No","Tanggal","Komentar","Komentar_Bersih","Sentimen","Kepercayaan"]].head(8), use_container_width=True)
    with right:
        st.write("Ringkasan statistik singkat")
        stats = pd.DataFrame({
            "Panjang (kata)": df_display["Komentar_Bersih"].str.split().map(len),
            "Kepercayaan": df_display.get("Kepercayaan", pd.Series([np.nan]*len(df_display)))
        }).describe().T
        st.dataframe(stats, use_container_width=True, height=220)

    # download
    buff = io.StringIO()
    df_display.to_csv(buff, index=False)
    st.download_button("⬇️ Download CSV Hasil Analisis", data=buff.getvalue(), file_name="hasil_analisis_sentimen.csv", mime="text/csv")

def page_analisis(df_display):
    st.subheader("Analisis — Distribusi & WordCloud")
    dist = df_display["Sentimen"].value_counts().reset_index()
    dist.columns = ["Sentimen","Jumlah"]
    fig_pie = px.pie(dist, names="Sentimen", values="Jumlah", hole=0.35,
                     color_discrete_map={"positif":"#2ECC71","netral":ACCENT,"negatif":"#E74C3C"})
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_bar = px.bar(dist.sort_values("Jumlah"), x="Sentimen", y="Jumlah", text_auto=True,
                     color="Sentimen", color_discrete_map={"positif":"#2ECC71","netral":ACCENT,"negatif":"#E74C3C"})
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Word Cloud per Sentimen")
    c1,c2,c3 = st.columns(3)
    for label, col in zip(["positif","netral","negatif"], [c1,c2,c3]):
        text = " ".join(df_display.loc[df_display["Sentimen"]==label, "Komentar_Bersih"].astype(str))
        if not text.strip():
            col.info(f"Tidak ada teks untuk {label}.")
            continue
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig = plt.figure(figsize=(6,3.2))
        plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
        col.pyplot(fig, use_container_width=True)

def page_visual(df_display):
    st.subheader("Visualisasi — Trend & Scatter")
    tmp = df_display.copy()
    tmp["Tanggal_dt"] = pd.to_datetime(tmp["Tanggal"])
    trend = tmp.groupby([tmp["Tanggal_dt"].dt.to_period("D").astype(str), "Sentimen"]).size().reset_index(name="Jumlah")
    trend.columns = ["Tanggal","Sentimen","Jumlah"]
    fig_line = px.line(trend, x="Tanggal", y="Jumlah", color="Sentimen", markers=True,
                       color_discrete_map={"positif":"#2ECC71","netral":ACCENT,"negatif":"#E74C3C"})
    st.plotly_chart(fig_line, use_container_width=True)

    # scatter: panjang vs confidence
    tmp["Panjang"] = tmp["Komentar_Bersih"].str.split().map(len)
    # trendline only if statsmodels available
    try:
        import statsmodels.api  # noqa
        trendline_opt = "ols" if len(tmp) >= 10 else None
    except Exception:
        trendline_opt = None

    fig_sc = px.scatter(tmp, x="Panjang", y="Kepercayaan", color="Sentimen",
                        hover_data=["No","Tanggal","Komentar"], trendline=trendline_opt,
                        color_discrete_map={"positif":"#2ECC71","netral":ACCENT,"negatif":"#E74C3C"})
    st.plotly_chart(fig_sc, use_container_width=True)

    with st.expander("Tabel hasil (filterable)"):
        filt = st.multiselect("Pilih Sentimen", options=["positif","netral","negatif"], default=["positif","netral","negatif"])
        tab = tmp[tmp["Sentimen"].isin(filt)]
        st.dataframe(tab[["No","Tanggal","Komentar","Sentimen","Kepercayaan"]], use_container_width=True, height=360)

# route pages
if page == "Overview":
    page_overview(df_proc)
elif page == "Analisis":
    page_analisis(df_proc)
else:
    page_visual(df_proc)

# footer
st.markdown(f"<div style='text-align:center; padding:10px; opacity:.7;'>Created with ❤️ — Theme {PRIMARY} / {ACCENT} — Real-time Google Sheets</div>", unsafe_allow_html=True)

