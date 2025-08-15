# dashboard_interaktif.py
# Premium Sentiment Dashboard — Real-time Google Sheets + IndoBERT
# - Modern blue–purple theme, glass cards, blurred background image
# - Read Google Sheets CSV (real-time), optional local file upload
# - Preprocess (Bahasa Indonesia), inference IndoBERT (3 labels)
# - Cache predictions by CSV bytes (avoid re-run if unchanged)
# - Fast mode for demo (no model download)
#
# Usage:
# 1) pip install -r requirements.txt
# 2) streamlit run dashboard_interaktif.py
# ================================================================

import io
import re
import base64
import hashlib
from typing import List, Tuple
from datetime import date

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

# Transformers & torch
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional statsmodels for trendline
try:
    import statsmodels.api  # noqa: F401
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# ----------------- Configuration (Google Sheet ID you provided) -----------------
GSHEET_ID = "1VL8FwJrAAZHqEDErlkhPtQ-a69O4JcRkOBmnYPKH2PY"
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/export?format=csv"

# ----------------- Page config -----------------
st.set_page_config(page_title="Sentiment Dashboard — Premium",
                   page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ----------------- Theme & CSS (glass, blur, typography) -----------------
PRIMARY = "#5D3FD3"      # purple-blue
ACCENT = "#7FB3D5"       # soft blue accent
CARD_BG = "rgba(255,255,255,0.85)"
TEXT = "#0b1220"

st.markdown(f"""
<style>
:root {{
  --primary: {PRIMARY};
  --accent: {ACCENT};
  --card-bg: {CARD_BG};
  --text: {TEXT};
}}
/* typography */
body, .css-1d391kg, .stApp, .css-1v3fvcr {{
  font-family: Inter, Poppins, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  color: var(--text);
}}

/* hero */
.header-hero {{
  background: linear-gradient(90deg, rgba(93,63,211,0.06), rgba(127,179,213,0.03));
  padding: 18px; border-radius: 12px; margin-bottom: 14px;
  box-shadow: 0 8px 24px rgba(11,22,39,0.04);
}}
.header-hero h1 {{ color: var(--primary); margin: 0; font-weight:800; font-size:30px; }}
.header-hero p {{ margin:6px 0 0; color: #475569; }}

/* glass card */
.card {{
  background: var(--card-bg); border-radius: 12px; padding: 12px;
  box-shadow: 0 12px 30px rgba(10,20,30,0.06);
}}
.metric-title {{ font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.08em; }}
.metric-value {{ font-size:22px; font-weight:800; color:var(--text); }}

.stDownloadButton>button, .stButton>button {{
  border-radius:10px !important; padding:9px 14px !important;
  background: var(--primary) !important; color: #fff !important; border: none !important;
}}

/* background image element (blurred) */
.bg-image {{
  position: fixed; inset: 0; z-index: -2; background-position:center; background-size:cover;
  filter: blur(8px) brightness(.52); transform: scale(1.02);
}}
.overlay-tint {{
  position: fixed; inset: 0; z-index: -1;
  background: linear-gradient(135deg, rgba(93,63,211,0.30), rgba(34,103,178,0.22));
  mix-blend-mode: multiply;
  pointer-events: none;
}}
.content-wrap {{ position: relative; z-index: 1; }}
/* responsive */
@media (max-width: 768px) {{
  .header-hero h1 {{ font-size: 22px; }}
  .metric-value {{ font-size: 18px; }}
}}
</style>
""", unsafe_allow_html=True)

# ----------------- Helpers: text preprocessing (Indonesian) -----------------
# try Sastrawi stopwords
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    factory = StopWordRemoverFactory()
    STOPWORDS = set(factory.get_stop_words())
except Exception:
    STOPWORDS = set()

def compress_repeats(s: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1\1', s)

def clean_text(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'@[^\s]+|#[^\s]+', ' ', t)
    t = t.encode('ascii', 'ignore').decode('ascii')   # strip emojis (keeps ascii)
    t = re.sub(r'[^a-z\s]', ' ', t)
    t = compress_repeats(t)
    if STOPWORDS:
        tokens = [w for w in t.split() if w and w not in STOPWORDS]
    else:
        tokens = [w for w in t.split() if w]
    return " ".join(tokens).strip()

# ----------------- Model loader (cached) -----------------
@st.cache_resource(show_spinner=True)
def load_indobert_model():
    PRETRAIN = "mdhugol/indonesia-bert-sentiment-classification"
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAIN)
    label_map = {"LABEL_0": "positif", "LABEL_1": "netral", "LABEL_2": "negatif"}
    return tokenizer, model, label_map

# ----------------- Prediction w/ caching by CSV bytes -----------------
@st.cache_data(show_spinner=False)
def predict_from_bytes(csv_bytes: bytes, fast_mode: bool, batch_size: int = 32) -> pd.DataFrame:
    bio = io.BytesIO(csv_bytes)
    df = pd.read_csv(bio)
    # normalize column names (strip) and map common names case-insensitively
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for expected in ["No","Tanggal","Komentar"]:
        for c in df.columns:
            if c.strip().lower() == expected.lower():
                col_map[c] = expected
    if col_map:
        df = df.rename(columns=col_map)
    # validate
    if not set(["No","Tanggal","Komentar"]).issubset(set(df.columns)):
        return pd.DataFrame({"error":["Missing required columns: No, Tanggal, Komentar"]})
    df = df.dropna(subset=["Komentar"]).reset_index(drop=True)
    # preprocessing
    df["Komentar_Bersih"] = df["Komentar"].astype(str).apply(clean_text)
    if fast_mode:
        rng = np.random.default_rng(2025)
        df["Sentimen"] = rng.choice(["positif","netral","negatif"], size=len(df))
        df["Kepercayaan"] = (rng.random(len(df))*0.4 + 0.6).round(4)
        return df
    # real inference
    tokenizer, model, mapping = load_indobert_model()
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

# ----------------- Sidebar: data & settings -----------------
with st.sidebar:
    st.title("📂 Sumber Data & Pengaturan")
    st.write("Data diambil real-time dari Google Sheets (link yang kamu berikan).")
    mode = st.selectbox("Mode data", ["Google Sheets (real-time)", "Local file (CSV/XLSX)"])
    uploaded = None
    if mode == "Local file (CSV/XLSX)":
        uploaded = st.file_uploader("Upload file lokal", type=["csv","xlsx"])
    st.markdown("---")
    st.subheader("⚙️ Analisis")
    fast_mode = st.toggle("Fast mode (no model) — cepat demo", value=False)
    sample_limit = st.slider("Batas baris yang diproses (demo)", 50, 2000, 1000, step=50)
    st.markdown("---")
    st.subheader("🎨 Background & Tema")
    bg_upload = st.file_uploader("Upload background image (jpg/png) — optional", type=["png","jpg","jpeg"])
    overlay_opacity = st.slider("Overlay opacity", 0.0, 0.85, 0.45, 0.05)
    st.markdown("---")
    st.caption("Fast mode = preview cepat tanpa mengunduh model. Non-fast mode akan mengunduh model pada first run.")

# ----------------- Fetch source bytes (sheet or uploaded) -----------------
@st.cache_data(ttl=60)
def fetch_csv_bytes_from_url(url: str) -> bytes:
    df_temp = pd.read_csv(url)
    bio = io.BytesIO()
    df_temp.to_csv(bio, index=False)
    return bio.getvalue()

raw_bytes = None
if mode == "Google Sheets (real-time)":
    try:
        raw_bytes = fetch_csv_bytes_from_url(GSHEET_CSV_URL)
    except Exception as e:
        st.sidebar.error("Gagal membaca Google Sheets: " + str(e))
        st.stop()
else:
    if uploaded is None:
        st.sidebar.info("Belum mengunggah file lokal.")
        st.stop()
    else:
        if uploaded.name.lower().endswith(".xlsx"):
            df_local = pd.read_excel(uploaded)
            bio = io.BytesIO(); df_local.to_csv(bio, index=False); raw_bytes = bio.getvalue()
        else:
            raw_bytes = uploaded.read()

# ----------------- Background: show uploaded image (blur + overlay) -----------------
if bg_upload is not None:
    raw_bg = bg_upload.read()
    b64 = base64.b64encode(raw_bg).decode()
    st.markdown(f'<div class="bg-image" style="background-image: url(data:image/png;base64,{b64});"></div>', unsafe_allow_html=True)
else:
    # default subtle gradient background
    st.markdown(f'<div class="bg-image" style="background: linear-gradient(120deg, rgba(93,63,211,0.18), rgba(34,103,178,0.12));"></div>', unsafe_allow_html=True)
# overlay tint to improve contrast
st.markdown(f'<div class="overlay-tint" style="opacity:{overlay_opacity};"></div>', unsafe_allow_html=True)

# ----------------- Predict (cache-aware) -----------------
with st.spinner("🔄 menyiapkan data & inferensi (cache-aware)..."):
    df_all = predict_from_bytes(raw_bytes, fast_mode=fast_mode, batch_size=32)

if "error" in df_all.columns:
    st.error(df_all["error"].iat[0])
    st.stop()

# normalize & parse
df_all.columns = [c.strip() for c in df_all.columns]
if "Tanggal" in df_all.columns:
    df_all["Tanggal"] = pd.to_datetime(df_all["Tanggal"], errors="coerce").dt.date
df_all = df_all.dropna(subset=["Komentar"]).reset_index(drop=True)

# apply sample limit
df_proc = df_all.head(min(len(df_all), sample_limit)).copy()

# ----------------- Header -----------------
st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
st.markdown('<div class="header-hero"><h1>🧠 Sentiment Dashboard — Premium</h1><p>Realtime Google Sheets • IndoBERT • Modern UI • Background blur</p></div>', unsafe_allow_html=True)

# ----------------- Filter panel -----------------
with st.expander("🔧 Filter Data (global)", expanded=True):
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if "Tanggal" in df_proc.columns:
            dmin = pd.to_datetime(df_proc["Tanggal"]).min().date()
            dmax = pd.to_datetime(df_proc["Tanggal"]).max().date()
            drange = st.date_input("Rentang tanggal", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        else:
            drange = None
    with col2:
        keyword = st.text_input("Cari kata (komentar)", value="")
    with col3:
        candidate_cols = [c for c in df_proc.columns if df_proc[c].dtype == object and c.lower() not in ("komentar","tanggal")]
        cat_col = st.selectbox("Filter kolom kategori (opsional)", options=["(none)"] + candidate_cols)
        cat_vals = None
        if cat_col and cat_col != "(none)":
            opts = sorted(df_proc[cat_col].dropna().unique().tolist())
            cat_vals = st.multiselect(f"Pilih {cat_col}", options=opts, default=opts[:6] if len(opts)>6 else opts)

# apply filters
df_view = df_proc.copy()
if drange and isinstance(drange, tuple) and len(drange) == 2:
    s,e = drange
    ser = pd.to_datetime(df_view["Tanggal"]).dt.date
    df_view = df_view[(ser >= s) & (ser <= e)]
if keyword:
    df_view = df_view[df_view["Komentar"].str.contains(keyword, case=False, na=False)]
if cat_vals:
    df_view = df_view[df_view[cat_col].isin(cat_vals)]

if df_view.empty:
    st.warning("Tidak ada data setelah filter. Coba ubah filter.")
    st.stop()

# ----------------- Metrics -----------------
total = len(df_view)
cnt_pos = int((df_view["Sentimen"] == "positif").sum())
cnt_neu = int((df_view["Sentimen"] == "netral").sum())
cnt_neg = int((df_view["Sentimen"] == "negatif").sum())
avg_conf = float(df_view["Kepercayaan"].mean()) if "Kepercayaan" in df_view.columns else np.nan

m1,m2,m3,m4 = st.columns(4)
with m1: st.markdown(f'<div class="card"><div class="metric-title">Total (terfilter)</div><div class="metric-value">{total:,}</div></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="card"><div class="metric-title">Positif</div><div class="metric-value">{cnt_pos:,}</div></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="card"><div class="metric-title">Netral</div><div class="metric-value">{cnt_neu:,}</div></div>', unsafe_allow_html=True)
with m4: st.markdown(f'<div class="card"><div class="metric-title">Negatif</div><div class="metric-value">{cnt_neg:,}</div></div>', unsafe_allow_html=True)

st.markdown(f"**Insight singkat:** Dominan **{('Positif' if cnt_pos>=max(cnt_neu,cnt_neg) else ('Netral' if cnt_neu>=cnt_neg else 'Negatif'))}** • Rata-rata confidence: **{avg_conf*100:.1f}%**.")

# representative comments
def sample_examples(dfm: pd.DataFrame, label: str, n: int = 3):
    sub = dfm[dfm["Sentimen"]==label]
    if sub.empty: return ["-"]
    return sub["Komentar"].sample(n=min(len(sub), n), random_state=42).tolist()

examples_pos = sample_examples(df_view, "positif", 3)
examples_neu = sample_examples(df_view, "netral", 3)
examples_neg = sample_examples(df_view, "negatif", 3)

# ----------------- Navigation -----------------
if HAS_OPTION_MENU:
    page = option_menu(None, ["Overview","Analisis","Visualisasi"], icons=["house","chat","bar-chart"], default_index=0, orientation="horizontal")
else:
    page = st.radio("Halaman", ["Overview","Analisis","Visualisasi"], horizontal=True)

# ----------------- Pages -----------------
def page_overview():
    st.subheader("Overview — Preview & Download")
    left, right = st.columns([2,1])
    with left:
        cols_show = [c for c in ["No","Tanggal","Komentar","Komentar_Bersih","Sentimen","Kepercayaan"] if c in df_view.columns]
        st.dataframe(df_view[cols_show].head(12), use_container_width=True, height=320)
    with right:
        st.markdown("**Ringkasan**")
        stats = pd.DataFrame({
            "Panjang (kata)": df_view["Komentar_Bersih"].str.split().map(len),
            "Kepercayaan": df_view.get("Kepercayaan", pd.Series([np.nan]*len(df_view)))
        }).describe().T
        st.dataframe(stats, use_container_width=True, height=260)
        st.markdown("**Contoh komentar**")
        st.markdown(f"- Positif: {examples_pos[0] if examples_pos else '-'}")
        st.markdown(f"- Netral: {examples_neu[0] if examples_neu else '-'}")
        st.markdown(f"- Negatif: {examples_neg[0] if examples_neg else '-'}")

    buff = io.StringIO(); df_view.to_csv(buff, index=False)
    st.download_button("⬇️ Download CSV Hasil Analisis", data=buff.getvalue(), file_name="hasil_analisis_sentimen.csv", mime="text/csv")

def page_analisis():
    st.subheader("Analisis — Distribusi & Keywords")
    dist = df_view["Sentimen"].value_counts().reset_index(); dist.columns = ["Sentimen","Jumlah"]
    c1,c2 = st.columns([1,1])
    with c1:
        fig = px.pie(dist, names="Sentimen", values="Jumlah", hole=0.36,
                     color_discrete_map={"positif":"#7AE582","netral":"#FFD97A","negatif":"#FF8A8A"})
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(dist.sort_values("Jumlah", ascending=False), x="Sentimen", y="Jumlah", text_auto=True,
                      color="Sentimen", color_discrete_map={"positif":"#7AE582","netral":"#FFD97A","negatif":"#FF8A8A"})
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")
    for label in ["positif","netral","negatif"]:
        text = " ".join(df_view.loc[df_view["Sentimen"]==label, "Komentar_Bersih"].astype(str))
        if not text.strip():
            st.info(f"No data for {label}")
            continue
        words = pd.Series(text.split()).value_counts().head(8)
        st.markdown(f"**{label.capitalize()} top words:** " + ", ".join([f"{w} ({cnt})" for w,cnt in words.items()]))
    st.markdown("#### Wordclouds")
    col1,col2,col3 = st.columns(3)
    for label, col in zip(["positif","netral","negatif"], [col1,col2,col3]):
        text = " ".join(df_view.loc[df_view["Sentimen"]==label, "Komentar_Bersih"].astype(str))
        if not text.strip():
            col.info(label)
            continue
        wc = WordCloud(width=900, height=400, background_color="white", max_words=80).generate(text)
        fig = plt.figure(figsize=(6,3))
        plt.imshow(wc); plt.axis("off")
        col.pyplot(fig, use_container_width=True)

def page_visual():
    st.subheader("Visualisasi — Trend & Confidence")
    if "Tanggal" in df_view.columns:
        tmp = df_view.copy(); tmp["Tanggal_dt"] = pd.to_datetime(tmp["Tanggal"])
        trend = tmp.groupby([tmp["Tanggal_dt"].dt.to_period("D").astype(str), "Sentimen"]).size().reset_index(name="Jumlah")
        trend.columns = ["Tanggal","Sentimen","Jumlah"]
        fig = px.line(trend, x="Tanggal", y="Jumlah", color="Sentimen", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Kolom Tanggal tidak tersedia.")
    tmp2 = df_view.copy(); tmp2["Panjang"] = tmp2["Komentar_Bersih"].str.split().map(len)
    trendline_opt = "ols" if HAS_STATSMODELS and len(tmp2) >= 10 else None
    fig2 = px.scatter(tmp2, x="Panjang", y="Kepercayaan", color="Sentimen", hover_data=["No","Tanggal","Komentar"], trendline=trendline_opt)
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("📋 Tabel hasil (filterable)"):
        sel = st.multiselect("Filter Sentimen", options=["positif","netral","negatif"], default=["positif","netral","negatif"])
        tab = tmp2[tmp2["Sentimen"].isin(sel)]
        st.dataframe(tab[["No","Tanggal","Komentar","Sentimen","Kepercayaan"]], use_container_width=True, height=360)
        buf = io.StringIO(); tab.to_csv(buf, index=False)
        st.download_button("💾 Download CSV (ter-filter)", data=buf.getvalue(), file_name="hasil_filter_sentimen.csv", mime="text/csv")

# route pages
if page == "Overview":
    page_overview()
elif page == "Analisis":
    page_analisis()
else:
    page_visual()

st.markdown("</div>", unsafe_allow_html=True)  # close content wrap
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:12px; opacity:.7;'>Created with ❤️ — Premium UI • IndoBERT • Real-time Google Sheets</div>", unsafe_allow_html=True)
