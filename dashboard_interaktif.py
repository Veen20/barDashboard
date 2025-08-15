# dashboard_interaktif.py
# Total makeover dashboard — glassmorphism, parallax blurred background,
# floating nav, modern fonts, IndoBERT sentiment, realtime Google Sheets.
#
# Save as: dashboard_interaktif.py
# Run:
#   pip install -r requirements.txt
#   streamlit run dashboard_interaktif.py
#
# Expected sheet columns (case-insensitive): No | Tanggal | Komentar
# --------------------------------------------------------------------

import io, re, base64, textwrap, hashlib
from typing import List, Tuple
from datetime import date
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Optional nicer menu
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# Transformers & torch
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# statsmodels optional for trendline
try:
    import statsmodels.api  # noqa
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# ----------------- CONFIG (Use your Google Sheet ID) -----------------
GSHEET_ID = "1VL8FwJrAAZHqEDErlkhPtQ-a69O4JcRkOBmnYPKH2PY"
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/export?format=csv"

# ----------------- STREAMLIT PAGE CONFIG -----------------
st.set_page_config(page_title="Samsat Sentiment — Premium UI",
                   page_icon="🚀", layout="wide", initial_sidebar_state="collapsed")

# ----------------- CSS: full redesign (glass, parallax bg, floating nav) -----------------
# Note: we hide default Streamlit top bar and footer for cleaner app-look
css = r"""
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Poppins', system-ui, -apple-system, "Segoe UI", Roboto, Arial !important; }

/* hide default streamlit header and footer */
header, footer { visibility: hidden; height: 0px; }

/* full-screen background image (parallax-like) */
.bg-cover{
  position: fixed;
  inset: 0;
  z-index: -99;
  background-position: center;
  background-size: cover;
  transform: scale(1.03);
  filter: blur(10px) brightness(.48) saturate(.9);
}

/* overlay gradient tint */
.bg-tint{
  position: fixed;
  inset: 0;
  z-index: -90;
  background: linear-gradient(120deg, rgba(93,63,211,0.35), rgba(35,86,163,0.22));
  mix-blend-mode: multiply;
}

/* content container */
.app-container { position: relative; z-index: 2; padding: 28px; }

/* floating left nav */
.floating-nav {
  position: fixed;
  left: 20px;
  top: 20px;
  z-index: 30;
  background: rgba(255,255,255,0.06);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 8px;
  border-radius: 12px;
  box-shadow: 0 10px 30px rgba(2,6,23,0.30);
}
.floating-nav a { color: #fff; text-decoration: none; display:block; padding:8px 12px; border-radius:8px; margin:6px 0; }
.floating-nav a:hover { background: rgba(255,255,255,0.06); transform: translateX(4px); transition: 0.18s; }

/* hero */
.hero {
  display:flex; gap:18px; align-items:center; justify-content:space-between;
  padding:18px; border-radius:14px;
  background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.04);
  box-shadow: 0 12px 36px rgba(2,6,23,0.25);
}
.hero-left { display:flex; gap:14px; align-items:center; }
.logo-badge { width:56px; height:56px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-weight:800; color:#fff; background: linear-gradient(135deg,#5D3FD3,#2E86C1); box-shadow: 0 8px 20px rgba(46,30,90,0.28); }
.hero h1 { margin:0; color: #EAF2FF; font-size:28px; letter-spacing:-0.02em; }
.hero p { margin:0; color: rgba(240,245,255,0.85); font-size:13px; opacity:0.9; }

/* glass card */
.card {
  background: rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 12px;
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 10px 30px rgba(7,12,30,0.40);
  backdrop-filter: blur(8px) saturate(.9);
  color: #EAF2FF;
}
.metric-title { font-size:11px; color: rgba(255,255,255,0.7); text-transform:uppercase; letter-spacing:.08em; }
.metric-value { font-size:22px; font-weight:800; color: #fff; }

/* panels */
.section { margin-top:18px; margin-bottom:18px; }

/* table styling hint */
div[data-testid="stDataFrame"] table { background: rgba(255,255,255,0.04); color: #fff; }

/* small screens */
@media (max-width: 768px) {
  .floating-nav { left:10px; top:10px; padding:6px; }
  .hero h1 { font-size:20px; }
}
"""
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ----------------- Background image handling -----------------
# prefer local samsat.jpg (user provided). If not present, use default gradient.
def get_bg_base64():
    # check for samsat.jpg in current dir
    if os.path.exists("samsat.jpg"):
        with open("samsat.jpg", "rb") as f:
            data = f.read()
            return base64.b64encode(data).decode()
    # (optional) embedded default small gradient image (1x1) fallback
    return None

bg_b64 = get_bg_base64()

if bg_b64:
    st.markdown(f'<div class="bg-cover" style="background-image: url(data:image/jpeg;base64,{bg_b64});"></div>', unsafe_allow_html=True)
else:
    # gradient fallback
    st.markdown('<div class="bg-cover" style="background: linear-gradient(120deg,#5D3FD3,#2E86C1);"></div>', unsafe_allow_html=True)

# tint overlay
st.markdown('<div class="bg-tint"></div>', unsafe_allow_html=True)

# ----------------- Helpers: text cleaning -----------------
# Use Sastrawi stopwords if available
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    factory = StopWordRemoverFactory()
    STOPWORDS = set(factory.get_stop_words())
except Exception:
    STOPWORDS = set()

def compress_repeats(s: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1\1', s)

def clean_text(txt: str) -> str:
    t = str(txt).lower()
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'@[^\s]+|#[^\s]+', ' ', t)
    # remove non-ascii (strip emojis)
    t = t.encode('ascii', 'ignore').decode('ascii')
    t = re.sub(r'[^a-z\s]', ' ', t)
    t = compress_repeats(t)
    toks = [w for w in t.split() if w and w not in STOPWORDS]
    return " ".join(toks).strip()

# ----------------- Model loader (cached) -----------------
@st.cache_resource(show_spinner=True)
def load_model():
    PRETRAIN = "mdhugol/indonesia-bert-sentiment-classification"
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAIN)
    label_map = {"LABEL_0":"positif","LABEL_1":"netral","LABEL_2":"negatif"}
    return tokenizer, model, label_map

# ----------------- Prediction cached by CSV bytes -----------------
@st.cache_data(show_spinner=False)
def predict_from_bytes(csv_bytes: bytes, fast_mode: bool, batch_size: int = 32) -> pd.DataFrame:
    bio = io.BytesIO(csv_bytes)
    df = pd.read_csv(bio)
    df.columns = [c.strip() for c in df.columns]
    # try map common names case-insensitively
    colmap = {}
    for want in ["No","Tanggal","Komentar"]:
        for c in df.columns:
            if c.strip().lower() == want.lower():
                colmap[c] = want
    if colmap:
        df = df.rename(columns=colmap)
    if not set(["No","Tanggal","Komentar"]).issubset(set(df.columns)):
        return pd.DataFrame({"error":["Missing required columns: No, Tanggal, Komentar"]})
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
        enc = {k:v.to(device) for k,v in enc.items()}
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

# ----------------- Controls (floating nav & control panel) -----------------
# Build floating nav markup
nav_html = """
<div class="floating-nav">
  <a href="#overview">🏠 Overview</a>
  <a href="#analisis">📊 Analisis</a>
  <a href="#visualisasi">📈 Visualisasi</a>
  <a href="#download">⬇️ Export</a>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# Right-side "control strip" implemented with sidebar for inputs (kept hidden visually)
with st.sidebar:
    st.header("Pengaturan")
    st.markdown("Sumber data & mode")
    mode = st.selectbox("Mode data", ["Google Sheets (real-time)", "Local file (CSV/XLSX)"])
    uploaded = None
    if mode == "Local file (CSV/XLSX)":
        uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])
    st.markdown("---")
    st.subheader("Analisis")
    fast_mode = st.toggle("Fast mode (no model) — demo cepat", value=False)
    sample_limit = st.slider("Batas baris diproses (demo)", min_value=50, max_value=5000, value=1000, step=50)
    st.markdown("---")
    st.subheader("Background")
    bg_file = st.file_uploader("Upload background image (optional)", type=["jpg","jpeg","png"])
    overlay_opacity = st.slider("Overlay opacity", min_value=0.0, max_value=0.9, value=0.46, step=0.02)
    st.markdown("---")
    st.caption("Tip: untuk demo cepat aktifkan Fast mode; save button tersedia di bagian Export.")

# ----------------- Load data bytes (sheet or file) -----------------
@st.cache_data(ttl=45)
def fetch_csv_bytes(url: str) -> bytes:
    df_tmp = pd.read_csv(url)
    bio = io.BytesIO(); df_tmp.to_csv(bio, index=False)
    return bio.getvalue()

raw_bytes = None
if mode == "Google Sheets (real-time)":
    try:
        raw_bytes = fetch_csv_bytes(GSHEET_CSV_URL)
    except Exception as e:
        st.sidebar.error("Gagal ambil Google Sheet: " + str(e))
        st.stop()
else:
    if uploaded is None:
        st.sidebar.info("Belum pilih file lokal.")
        st.stop()
    else:
        if uploaded.name.lower().endswith(".xlsx"):
            df_local = pd.read_excel(uploaded)
            b = io.BytesIO(); df_local.to_csv(b, index=False); raw_bytes = b.getvalue()
        else:
            raw_bytes = uploaded.read()

# if user provided bg_file, show it (overrides default)
if bg_file is not None:
    raw_bg = bg_file.read()
    b64 = base64.b64encode(raw_bg).decode()
    st.markdown(f'<div class="bg-cover" style="background-image:url(data:image/png;base64,{b64});"></div>', unsafe_allow_html=True)

# tint overlay with user opacity
st.markdown(f'<div class="bg-tint" style="opacity:{overlay_opacity};"></div>', unsafe_allow_html=True)

# ----------------- Run inference (cache-aware) -----------------
with st.spinner("Menyiapkan data dan inferensi (cache-aware)..."):
    df_all = predict_from_bytes(raw_bytes, fast_mode=fast_mode, batch_size=32)

if "error" in df_all.columns:
    st.error(df_all["error"].iat[0])
    st.stop()

# normalize
df_all.columns = [c.strip() for c in df_all.columns]
if "Tanggal" in df_all.columns:
    df_all["Tanggal"] = pd.to_datetime(df_all["Tanggal"], errors="coerce").dt.date
df_all = df_all.dropna(subset=["Komentar"]).reset_index(drop=True)
df_proc = df_all.head(min(len(df_all), sample_limit)).copy()

# ----------------- Hero -----------------
st.markdown('<div class="app-container">', unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="hero-left"><div class="logo-badge">SP</div><div><h1 id="overview">Samsat Sentiment</h1><p>Realtime • IndoBERT • Premium UI</p></div></div></div>', unsafe_allow_html=True)

# ----------------- Filters (inline) -----------------
with st.expander("🔧 Filter cepat", expanded=True):
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if "Tanggal" in df_proc.columns:
            dmin = pd.to_datetime(df_proc["Tanggal"]).min().date()
            dmax = pd.to_datetime(df_proc["Tanggal"]).max().date()
            drange = st.date_input("Rentang tanggal", value=(dmin,dmax), min_value=dmin, max_value=dmax)
        else:
            drange = None
    with c2:
        keyword = st.text_input("Cari kata di komentar", value="")
    with c3:
        candidate = [c for c in df_proc.columns if df_proc[c].dtype == object and c.lower() not in ("komentar","tanggal")]
        cat_col = st.selectbox("Filter kategori (opsional)", options=["(none)"] + candidate)
        cat_vals = None
        if cat_col and cat_col != "(none)":
            opts = sorted(df_proc[cat_col].dropna().unique().tolist())
            cat_vals = st.multiselect(f"Pilih {cat_col}", options=opts, default=opts[:6] if len(opts)>6 else opts)

# apply filters
df_view = df_proc.copy()
if drange and isinstance(drange, tuple) and len(drange)==2:
    s,e = drange; ser = pd.to_datetime(df_view["Tanggal"]).dt.date
    df_view = df_view[(ser >= s) & (ser <= e)]
if keyword:
    df_view = df_view[df_view["Komentar"].str.contains(keyword, case=False, na=False)]
if cat_vals:
    df_view = df_view[df_view[cat_col].isin(cat_vals)]

if df_view.empty:
    st.warning("Tidak ada data setelah filter — coba ubah filter.")
    st.stop()

# ----------------- Metrics (stylish) -----------------
total = len(df_view)
cpos = int((df_view["Sentimen"]=="positif").sum())
cneu = int((df_view["Sentimen"]=="netral").sum())
cneg = int((df_view["Sentimen"]=="negatif").sum())
avg_conf = float(df_view["Kepercayaan"].mean()) if "Kepercayaan" in df_view.columns else np.nan

m1,m2,m3,m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="card"><div class="metric-title">Total (terfilter)</div><div class="metric-value">{total:,}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="card"><div class="metric-title">Positif</div><div class="metric-value">{cpos:,}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="card"><div class="metric-title">Netral</div><div class="metric-value">{cneu:,}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="card"><div class="metric-title">Negatif</div><div class="metric-value">{cneg:,}</div></div>', unsafe_allow_html=True)

st.markdown(f"**Insight singkat:** Dominan **{('Positif' if cpos>=max(cneu,cneg) else ('Netral' if cneu>=cneg else 'Negatif'))}** • Rata-rata confidence: **{avg_conf*100:.1f}%**.")

# representative examples
def sample_examples(dfm, label, n=3):
    sub = dfm[dfm["Sentimen"]==label]
    if sub.empty: return ["-"]
    return sub["Komentar"].sample(n=min(n,len(sub)), random_state=42).tolist()

ex_pos = sample_examples(df_view, "positif")
ex_neu = sample_examples(df_view, "netral")
ex_neg = sample_examples(df_view, "negatif")

# ----------------- Page navigation (tabs) -----------------
if HAS_OPTION_MENU:
    page = option_menu(None, ["Overview","Analisis","Visualisasi","Export"], icons=["house","graph-up","bar-chart","download"], default_index=0, orientation="horizontal")
else:
    page = st.selectbox("Halaman", ["Overview","Analisis","Visualisasi","Export"])

# ----------------- Pages -----------------
def page_overview():
    st.markdown("## Overview")
    left, right = st.columns([2,1])
    with left:
        cols = [c for c in ["No","Tanggal","Komentar","Komentar_Bersih","Sentimen","Kepercayaan"] if c in df_view.columns]
        st.dataframe(df_view[cols].head(12), use_container_width=True, height=340)
    with right:
        st.markdown("### Ringkasan Statistik")
        stats = pd.DataFrame({
            "Panjang (kata)": df_view["Komentar_Bersih"].str.split().map(len),
            "Kepercayaan": df_view.get("Kepercayaan", pd.Series([np.nan]*len(df_view)))
        }).describe().T
        st.dataframe(stats, use_container_width=True, height=260)
        st.markdown("### Contoh komentar")
        st.write("Positif:")
        for t in ex_pos: st.write("• " + t)
        st.write("Netral:")
        for t in ex_neu: st.write("• " + t)
        st.write("Negatif:")
        for t in ex_neg: st.write("• " + t)

def page_analisis():
    st.markdown("## Analisis")
    dist = df_view["Sentimen"].value_counts().reset_index(); dist.columns=["Sentimen","Jumlah"]
    a,b = st.columns([1,1])
    with a:
        fig = px.pie(dist, names="Sentimen", values="Jumlah", hole=0.36,
                     color_discrete_map={"positif":"#7AE582","netral":"#FFD97A","negatif":"#FF8A8A"})
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig2 = px.bar(dist.sort_values("Jumlah", ascending=False), x="Sentimen", y="Jumlah", text_auto=True,
                      color="Sentimen", color_discrete_map={"positif":"#7AE582","netral":"#FFD97A","negatif":"#FF8A8A"})
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")
    st.markdown("### Top Keywords per Label")
    for label in ["positif","netral","negatif"]:
        text = " ".join(df_view.loc[df_view["Sentimen"]==label, "Komentar_Bersih"].astype(str))
        if not text.strip():
            st.info(f"No data for {label}")
            continue
        topw = pd.Series(text.split()).value_counts().head(10)
        st.markdown(f"**{label.capitalize()}**: " + ", ".join([f"{w} ({c})" for w,c in topw.items()]))
    st.markdown("### Wordclouds")
    c1,c2,c3 = st.columns(3)
    for lbl,col in zip(["positif","netral","negatif"], [c1,c2,c3]):
        txt = " ".join(df_view.loc[df_view["Sentimen"]==lbl, "Komentar_Bersih"].astype(str))
        if not txt.strip():
            col.info(lbl)
            continue
        wc = WordCloud(width=800, height=360, background_color=None, mode="RGBA", colormap="viridis").generate(txt)
        fig = plt.figure(figsize=(6,3))
        plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
        col.pyplot(fig, use_container_width=True)

def page_visual():
    st.markdown("## Visualisasi")
    if "Tanggal" in df_view.columns:
        tmp = df_view.copy(); tmp["Tanggal_dt"] = pd.to_datetime(tmp["Tanggal"])
        trend = tmp.groupby([tmp["Tanggal_dt"].dt.to_period("D").astype(str), "Sentimen"]).size().reset_index(name="Jumlah")
        trend.columns = ["Tanggal","Sentimen","Jumlah"]
        fig = px.line(trend, x="Tanggal", y="Jumlah", color="Sentimen", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada kolom Tanggal.")
    tmp2 = df_view.copy(); tmp2["Panjang"] = tmp2["Komentar_Bersih"].str.split().map(len)
    trendline = "ols" if HAS_STATSMODELS and len(tmp2) >= 10 else None
    fig2 = px.scatter(tmp2, x="Panjang", y="Kepercayaan", color="Sentimen", hover_data=["No","Tanggal","Komentar"], trendline=trendline)
    st.plotly_chart(fig2, use_container_width=True)

def page_export():
    st.markdown("## Export / Save")
    st.markdown("Download hasil analisis (CSV)")
    buf = io.StringIO(); df_view.to_csv(buf, index=False)
    st.download_button("⬇️ Download CSV hasil analisis", data=buf.getvalue(), file_name="hasil_analisis_sentimen.csv", mime="text/csv")
    st.markdown("---")
    st.markdown("Pilihan lain:")
    st.write("- Gunakan Fast mode untuk demo cepat tanpa unduh model.")
    st.write("- Untuk auto-save ke Google Sheets diperlukan Service Account (bisa kubantu setup).")

# route
if page == "Overview":
    page_overview()
elif page == "Analisis":
    page_analisis()
elif page == "Visualisasi":
    page_visual()
else:
    page_export()

# footer end container
st.markdown("</div>", unsafe_allow_html=True)  # close app-container
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:rgba(255,255,255,0.7); padding-bottom:18px;'>Built with ❤️ — Premium UI • IndoBERT • Realtime Google Sheets</div>", unsafe_allow_html=True)
