# app.py — Dashboard Sentimen “Wah” (IndoBERT + Visual Premium)
# --------------------------------------------------------------
# Fitur:
# - UI modern (background + overlay + blur sidebar + animasi)
# - Data real-time Google Sheets / file lokal
# - IndoBERT sentiment (positif/netral/negatif) + confidence
# - Insight otomatis (dominasi sentimen, kata kunci, jam/harian)
# - Visual Plotly: distribusi, tren harian & jam, heatmap, sunburst
# - WordCloud per-sentimen
# - Unduh hasil (CSV) & filter fleksibel
# --------------------------------------------------------------
# Jalankan:
#   pip install -r requirements.txt
#   streamlit run app.py
# --------------------------------------------------------------

import io
import os
import re
import hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Optional enhancements
try:
    import statsmodels.api as sm  # trendline (opsional)
    HAS_SM = True
except Exception:
    HAS_SM = False

# -----------------------------
# Page Config (JANGAN duplikasi)
# -----------------------------
st.set_page_config(
    page_title="Dashboard Sentimen — IndoBERT (Wah!)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Load CSS (style.css di root)
# -----------------------------
def load_css(file_path="style.css"):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ style.css tidak ditemukan di: {file_path}")

load_css()

# -----------------------------
# Konstanta & Palet
# -----------------------------
PRIMARY = "#2E86C1"
ACCENT = "#F4D03F"
COLORS = {"positif": "#2ECC71", "netral": ACCENT, "negatif": "#E74C3C"}

# -----------------------------
# Sidebar: Sumber data & opsi
# -----------------------------
st.sidebar.title("⚙️ Pengaturan")
GSHEET_ID = st.sidebar.text_input("Google Sheets ID", value="1VL8FwJrAAZHqEDErlkhPtQ-a69O4JcRkOBmnYPKH2PY")
mode = st.sidebar.radio("Sumber Data", ["Google Sheets (real-time)", "File Lokal"])
uploaded = None
if mode == "File Lokal":
    uploaded = st.sidebar.file_uploader("Unggah CSV / XLSX", type=["csv", "xlsx"])

fast_mode = st.sidebar.toggle("Fast Mode (tanpa model) untuk demo cepat", value=False)
sample_limit = st.sidebar.slider("Batas baris (untuk percepatan)", 100, 10000, 2000, 100)
dark_mode = st.sidebar.toggle("Dark Mode", value=False)

# Dark mode (kelas CSS)
st.markdown(
    "<script>document.documentElement.classList.toggle('dark-mode', "
    + ("true" if dark_mode else "false") + ");</script>",
    unsafe_allow_html=True,
)

# -----------------------------
# Util: Preprocessing teks
# -----------------------------
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    STOPWORDS = set(StopWordRemoverFactory().get_stop_words())
except Exception:
    STOPWORDS = set()

def normalize_repeated_chars(s: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", s)

def clean_text(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)            # URL
    t = re.sub(r"@[^\s]+|#[^\s]+", " ", t)                  # mention/hashtag
    t = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", t)                # keep letters & digits
    t = normalize_repeated_chars(t)
    tokens = [w for w in t.split() if w and w not in STOPWORDS]
    return " ".join(tokens).strip()

# -----------------------------
# Model: IndoBERT Sentiment
# -----------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource(show_spinner=True)
def load_model():
    # Model yang stabil dan akurat untuk ID sentiment
    MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mapping = {"LABEL_0": "positif", "LABEL_1": "netral", "LABEL_2": "negatif"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, mapping, device

def predict_sentiment(texts: List[str], bsize: int = 32):
    tokenizer, model, mapping, device = load_model()
    labels, confs = [], []
    for i in range(0, len(texts), bsize):
        batch = texts[i:i+bsize]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=160, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            topv, topi = probs.max(dim=1)
        for idx, p in zip(topi.cpu().tolist(), topv.cpu().tolist()):
            labels.append(mapping.get(f"LABEL_{idx}", "netral"))
            confs.append(float(p))
    return labels, np.round(confs, 4)

# Cache berdasarkan snapshot CSV bytes
def df_to_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    return bio.getvalue()

@st.cache_data(show_spinner=False)
def infer_cached(csv_bytes: bytes, fast_mode: bool, bsize: int = 32):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    # Normalisasi nama kolom
    rename = {}
    for exp in ["No", "Tanggal", "Komentar"]:
        for c in df.columns:
            if c.strip().lower() == exp.lower():
                rename[c] = exp
    df = df.rename(columns=rename)
    if not set(["No", "Tanggal", "Komentar"]).issubset(df.columns):
        return pd.DataFrame({"error": ["Kolom wajib (No, Tanggal, Komentar) belum lengkap."]})

    df["Komentar"] = df["Komentar"].astype(str)
    df["Komentar_Bersih"] = df["Komentar"].apply(clean_text)

    if fast_mode:
        rng = np.random.default_rng(42)
        df["Sentimen"] = rng.choice(["positif", "netral", "negatif"], size=len(df))
        df["Kepercayaan"] = np.round(rng.random(len(df))*0.5 + 0.5, 4)
        return df

    labels, confs = predict_sentiment(df["Komentar_Bersih"].tolist(), bsize=bsize)
    df["Sentimen"] = labels
    df["Kepercayaan"] = confs
    return df

# -----------------------------
# Fetch data
# -----------------------------
@st.cache_data(ttl=60)
def fetch_gsheet_bytes(gid: str) -> bytes:
    url = f"https://docs.google.com/spreadsheets/d/{gid}/export?format=csv"
    df = pd.read_csv(url)
    return df_to_bytes(df)

if mode == "Google Sheets (real-time)":
    try:
        raw_bytes = fetch_gsheet_bytes(GSHEET_ID)
    except Exception as e:
        st.error(f"Gagal membaca Google Sheets: {e}")
        st.stop()
else:
    if uploaded is None:
        st.info("Silakan unggah file terlebih dahulu.")
        st.stop()
    raw_bytes = uploaded.read()

# -----------------------------
# Inference
# -----------------------------
with st.spinner("🚀 Memproses data & menjalankan IndoBERT..."):
    df = infer_cached(raw_bytes, fast_mode=fast_mode, bsize=32)

if "error" in df.columns:
    st.error(df["error"].iat[0])
    st.stop()

# Tanggal & subset
if "Tanggal" in df.columns:
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
else:
    df["Tanggal"] = pd.NaT

if sample_limit and len(df) > sample_limit:
    df = df.head(sample_limit).copy()

# -----------------------------
# HERO
# -----------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🧠 Dashboard Sentimen — IndoBERT (Wah!)</h1>
      <p>Real-time Google Sheets • Preprocessing otomatis • Visual interaktif • Insight siap presentasi</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Filter Global
# -----------------------------
with st.expander("🔎 Filter Data", expanded=True):
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        if df["Tanggal"].notna().any():
            dmin = df["Tanggal"].min().date()
            dmax = df["Tanggal"].max().date()
            daterng = st.date_input("Rentang Tanggal", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        else:
            daterng = None
    with c2:
        sent_filter = st.multiselect("Sentimen", ["positif","netral","negatif"], default=["positif","netral","negatif"])
    with c3:
        min_conf = st.slider("Batas Confidence (≥)", 0.0, 1.0, 0.50, 0.05)

dfv = df.copy()
if daterng and isinstance(daterng, tuple) and len(daterng) == 2 and dfv["Tanggal"].notna().any():
    s, e = pd.to_datetime(daterng[0]), pd.to_datetime(daterng[1])
    dfv = dfv[(dfv["Tanggal"] >= s) & (dfv["Tanggal"] <= e)]
dfv = dfv[dfv["Sentimen"].isin(sent_filter)]
dfv = dfv[dfv["Kepercayaan"] >= min_conf]

if dfv.empty:
    st.warning("Data kosong setelah filter. Coba longgarkan filter.")
    st.stop()

# -----------------------------
# METRIC CARDS + INSIGHT
# -----------------------------
total = len(dfv)
pos = int((dfv["Sentimen"] == "positif").sum())
neu = int((dfv["Sentimen"] == "netral").sum())
neg = int((dfv["Sentimen"] == "negatif").sum())
avg_conf = float(dfv["Kepercayaan"].mean())

m1, m2, m3, m4 = st.columns(4)
m1.markdown(f'<div class="metric-card"><div class="metric-title">Total</div><div class="metric-value">{total:,}</div></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="metric-card"><div class="metric-title">Positif</div><div class="metric-value">{pos:,}</div></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="metric-card"><div class="metric-title">Netral</div><div class="metric-value">{neu:,}</div></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="metric-card"><div class="metric-title">Negatif</div><div class="metric-value">{neg:,}</div></div>', unsafe_allow_html=True)

# Insight otomatis (singkat & to the point)
dominant = max([("positif", pos), ("netral", neu), ("negatif", neg)], key=lambda x: x[1])[0]
st.info(
    f"**Insight cepat:** Mayoritas komentar **{dominant}**. "
    f"Rata-rata kepercayaan model **{avg_conf*100:.1f}%**. "
    f"Rasio positif:negatif = **{(pos+1)/(neg+1):.2f}**."
)

# -----------------------------
# TABS: Overview | Analisis | Visual
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📋 Overview", "🔍 Analisis", "📈 Visual"])

# ====== TAB 1: OVERVIEW ======
with tab1:
    c1, c2 = st.columns([2,1])
    with c1:
        show_cols = [c for c in ["No","Tanggal","Komentar","Komentar_Bersih","Sentimen","Kepercayaan"] if c in dfv.columns]
        st.dataframe(dfv[show_cols], use_container_width=True, height=380)
    with c2:
        st.markdown("**Stat ringkas**")
        stats = pd.DataFrame({
            "Panjang (kata)": dfv["Komentar_Bersih"].str.split().map(len),
            "Kepercayaan": dfv["Kepercayaan"]
        }).describe().T
        st.dataframe(stats, use_container_width=True, height=220)

    # Unduh CSV
    buff = io.StringIO()
    dfv.to_csv(buff, index=False)
    st.download_button("⬇️ Unduh CSV Hasil", buff.getvalue(), "hasil_sentimen_indobert.csv", "text/csv")

# ====== TAB 2: ANALISIS ======
with tab2:
    st.subheader("Distribusi Sentimen")
    dist = dfv["Sentimen"].value_counts().reindex(["positif","netral","negatif"]).fillna(0).astype(int).reset_index()
    dist.columns = ["Sentimen","Jumlah"]
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(dist, names="Sentimen", values="Jumlah", hole=0.4,
                         color="Sentimen", color_discrete_map=COLORS)
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_bar = px.bar(dist, x="Sentimen", y="Jumlah", text_auto=True,
                         color="Sentimen", color_discrete_map=COLORS)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Kata Kunci Teratas (TF–IDF) per Sentimen")
    # TF-IDF sederhana tanpa library tambahan
    from collections import Counter
    def top_terms(df_sub, topn=12):
        txt = " ".join(df_sub["Komentar_Bersih"].tolist())
        words = [w for w in txt.split() if len(w) > 2]
        cnt = Counter(words)
        return pd.DataFrame(cnt.most_common(topn), columns=["term","freq"])

    cpos, cneu, cneg = st.columns(3)
    for label, col in zip(["positif","netral","negatif"], [cpos,cneu,cneg]):
        sub = dfv[dfv["Sentimen"]==label]
        if len(sub)==0:
            col.info(f"Tidak ada data **{label}**.")
            continue
        tt = top_terms(sub, 12)
        col.dataframe(tt, use_container_width=True, height=260)

    st.markdown("— *Gunakan daftar ini untuk menyusun rekomendasi kebijakan/layanan.*")

    st.markdown("---")
    st.subheader("WordCloud per Sentimen")
    c1, c2, c3 = st.columns(3)
    for label, col in zip(["positif","netral","negatif"], [c1,c2,c3]):
        text = " ".join(dfv.loc[dfv["Sentimen"]==label, "Komentar_Bersih"].tolist())
        if not text.strip():
            col.info(f"Belum ada teks untuk **{label}**.")
            continue
        wc = WordCloud(width=900, height=440, background_color="white", max_words=120).generate(text)
        fig = plt.figure(figsize=(6,3.1))
        plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
        col.pyplot(fig, use_container_width=True)

# ====== TAB 3: VISUAL ======
with tab3:
    st.subheader("Tren Harian")
    if dfv["Tanggal"].notna().any():
        tmp = dfv.copy()
        tmp["tgl"] = tmp["Tanggal"].dt.date
        trend = tmp.groupby(["tgl","Sentimen"]).size().reset_index(name="Jumlah")
        fig_line = px.line(trend, x="tgl", y="Jumlah", color="Sentimen", markers=True,
                           color_discrete_map=COLORS)
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Kolom tanggal belum valid untuk tren harian.")

    st.markdown("---")
    st.subheader("Pola Waktu (Heatmap by Jam)")
    if dfv["Tanggal"].notna().any():
        tmp = dfv.copy()
        tmp["jam"] = tmp["Tanggal"].dt.hour
        heat = tmp.groupby(["jam","Sentimen"]).size().reset_index(name="Jumlah")
        fig_hm = px.density_heatmap(heat, x="jam", y="Sentimen", z="Jumlah",
                                    nbinsx=24, color_continuous_scale="Blues")
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Tidak ada informasi jam; formatkan kolom tanggal ke datetime dengan waktu.")

    st.markdown("---")
    st.subheader("Hierarki (Sunburst) — Sentimen → (opsional) Kategori")
    # Jika ada kolom kategori (misal 'Channel' atau 'Loket'), kita pilih otomatis
    cat_cols = [c for c in dfv.columns if c.lower() not in ("no","tanggal","komentar","komentar_bersih","sentimen","kepercayaan")]
    if cat_cols:
        cat_col = st.selectbox("Pilih kolom kategori", options=cat_cols, index=0)
        sun = dfv.groupby(["Sentimen", cat_col]).size().reset_index(name="Jumlah")
        fig_sun = px.sunburst(sun, path=["Sentimen", cat_col], values="Jumlah", color="Sentimen",
                              color_discrete_map=COLORS)
        st.plotly_chart(fig_sun, use_container_width=True)
    else:
        st.info("Tidak ditemukan kolom kategori tambahan. Tambahkan kolom (mis. Channel/Loket) untuk analisis hierarki.")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "<footer>&copy; 2025 Dashboard Sentimen — IndoBERT • UI Premium • Dibuat untuk presentasi dosen</footer>",
    unsafe_allow_html=True
)
