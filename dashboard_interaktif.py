# =========================================================
# Dashboard Analisis Sentimen (IndoBERT) — untuk kolom:
# No | Tanggal | Komentar
# =========================================================
# Cara jalan:
#   pip install -r requirements.txt
#   streamlit run dashboard_interaktif.py
# =========================================================

import io
import re
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional (sidebar nav with icons). App tetap jalan kalau modul ini belum terpasang.
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# Visual tambahan
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# NLP (IndoBERT)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Konfigurasi halaman & tema
# -----------------------------
st.set_page_config(
    page_title="Dashboard Sentimen IndoBERT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#2E86C1"   # biru
ACCENT  = "#F4D03F"   # kuning
TEXT    = "#1B2631"
BG      = "#F8F9F9"

st.markdown(
    f"""
    <style>
      :root {{
        --primary: {PRIMARY};
        --accent: {ACCENT};
        --text: {TEXT};
        --bg: {BG};
      }}
      .hero {{
        background: linear-gradient(135deg, rgba(46,134,193,.10), rgba(244,208,63,.10));
        border: 1px solid rgba(0,0,0,.06);
        padding: 22px 24px;
        border-radius: 18px;
        margin-bottom: 14px;
        animation: fadeIn .4s ease;
      }}
      .hero h1 {{ color: var(--primary); margin: 0 0 6px 0; font-weight: 800; letter-spacing: -.02em; }}
      .hero p  {{ margin: 0; opacity: .85; font-size: 14px; }}
      .metric-card {{
        background: #fff; border: 1px solid rgba(0,0,0,.06);
        border-radius: 16px; padding: 14px;
        box-shadow: 0 8px 22px rgba(0,0,0,.05);
        transition: transform .15s ease, box-shadow .15s ease;
      }}
      .metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 12px 30px rgba(0,0,0,.08); }}
      .metric-title {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: #6c7a89; }}
      .metric-value {{ font-size: 24px; font-weight: 800; color: var(--text); }}
      .fade-in {{ animation: fadeIn .25s ease; }}
      @keyframes fadeIn {{ 0% {{opacity:0; transform: translateY(6px);}} 100% {{opacity:1; transform: translateY(0);}} }}
      /* tombol utama */
      .stDownloadButton button, .stButton>button {{
        border-radius: 12px !important; padding: 9px 14px !important;
        border: 1px solid rgba(0,0,0,.08) !important;
        background: var(--primary) !important; color: white !important;
      }}
      /* tema tabel */
      div[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Utilitas data & contoh
# -----------------------------
@st.cache_data
def example_data(n: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tanggal = pd.date_range("2025-01-01", periods=120, freq="D")
    sent = ["positif", "netral", "negatif"]
    komentar_bank = [
        "Pelayanan sangat cepat dan ramah",
        "Aplikasi sering error saat jam sibuk",
        "Biasa saja",
        "Prosesnya mudah dimengerti",
        "Terlalu lambat saat verifikasi",
        "Petugasnya informatif",
        "Kurang responsif",
        "Fitur lengkap dan membantu",
        "Susah diakses pada malam hari",
        "Mantap, sangat puas"
    ]
    df = pd.DataFrame({
        "No": np.arange(1, n + 1),
        "Tanggal": rng.choice(tanggal, n),
        "Komentar": rng.choice(komentar_bank, n),
    })
    # sisipkan beberapa noise
    df.loc[df.sample(frac=.05, random_state=seed).index, "Komentar"] += " 🙏🏻 #bagus"
    return df

# -----------------------------
# Preprocessing Bahasa Indonesia
# -----------------------------
# Sastrawi stopwords
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    _factory = StopWordRemoverFactory()
    STOPWORD = set(_factory.get_stop_words())
except Exception:
    STOPWORD = set()

def normalize_repeats(text: str) -> str:
    # "baaaagusss" -> "baagus"
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def clean_text(t: str) -> str:
    t = str(t)
    t = t.lower()
    # hapus url/mention/hashtag
    t = re.sub(r'(https?://\S+)|www\.\S+', ' ', t)
    t = re.sub(r'[@#]\w+', ' ', t)
    # hapus emoji & simbol non-ASCII
    t = t.encode('ascii', 'ignore').decode('ascii')
    # hapus angka & tanda baca non spasi
    t = re.sub(r'[^a-z\s]', ' ', t)
    # normalisasi huruf berulang
    t = normalize_repeats(t)
    # hapus stopword (jika tersedia)
    if STOPWORD:
        t = " ".join(w for w in t.split() if w not in STOPWORD)
    # rapikan spasi
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# -----------------------------
# Load model IndoBERT (3 kelas)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, dict]:
    """
    Model: mdhugol/indonesia-bert-sentiment-classification
    Mapping label dari model card:
        LABEL_0 -> positive
        LABEL_1 -> neutral
        LABEL_2 -> negative
    """
    pretrained = "mdhugol/indonesia-bert-sentiment-classification"
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained)
    label_index = {"LABEL_0": "positif", "LABEL_1": "netral", "LABEL_2": "negatif"}
    return tokenizer, model, label_index

tokenizer, model, LABEL_INDEX = load_model()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)
model.eval()

def predict_batch(texts: List[str], batch_size: int = 32) -> Tuple[List[str], List[float]]:
    labels, confs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=160,
            return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            top_p, top_i = probs.max(dim=1)
        # map ke label indo
        for idx, p in zip(top_i.cpu().tolist(), top_p.cpu().tolist()):
            # idx -> LABEL_n
            key = f"LABEL_{idx}"
            labels.append(LABEL_INDEX.get(key, "netral"))
            confs.append(float(p))
    return labels, confs

# -----------------------------
# Sidebar: data & navigasi
# -----------------------------
with st.sidebar:
    st.markdown("### 📂 Data & Navigasi")
    st.caption("Unggah CSV/XLSX dengan kolom **No, Tanggal, Komentar** atau pakai dataset contoh.")
    uploaded = st.file_uploader("Upload file", type=["csv", "xlsx"])
    use_sample = st.toggle("Gunakan dataset contoh", value=uploaded is None)

    if HAS_OPTION_MENU:
        page = option_menu(
            None,
            ["Overview", "Analisis Sentimen", "Visualisasi"],
            icons=["house", "chat-left-dots", "bar-chart"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background": "transparent"},
                "icon": {"color": PRIMARY, "font-size": "18px"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color":"#eee"},
                "nav-link-selected": {"background-color": ACCENT, "color": "#000"},
            },
        )
    else:
        page = st.radio("Pilih Halaman", ["Overview", "Analisis Sentimen", "Visualisasi"], index=0)

# -----------------------------
# Load data (upload / contoh)
# -----------------------------
def load_df() -> pd.DataFrame:
    if uploaded:
        if uploaded.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded)
        else:
            try:
                df = pd.read_csv(uploaded)
            except Exception:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, encoding="latin1")
    else:
        df = example_data()
    return df

df = load_df().copy()

# Validasi & perapihan kolom
expected = {"No", "Tanggal", "Komentar"}
missing = expected - set(df.columns)
if missing:
    st.error(f"Kolom wajib hilang: {', '.join(missing)}. Pastikan kolom tepat: **No, Tanggal, Komentar**.")
    st.stop()

# Tipe data
df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce").dt.date
df = df.dropna(subset=["Komentar"]).reset_index(drop=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🧠 Dashboard Analisis Sentimen (IndoBERT)</h1>
      <p>Preprocessing otomatis • Klasifikasi 3 kelas (positif / netral / negatif) • Visual interaktif • Download hasil.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Panel filter global
# -----------------------------
with st.expander("🔧 Filter Data (real-time)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        # Filter tanggal
        dmin = pd.to_datetime(df["Tanggal"]).min().date() if len(df) else None
        dmax = pd.to_datetime(df["Tanggal"]).max().date() if len(df) else None
        if dmin and dmax:
            f_start, f_end = st.date_input("Rentang Tanggal", (dmin, dmax), min_value=dmin, max_value=dmax, format="YYYY-MM-DD")
        else:
            f_start, f_end = None, None
    with c2:
        sample_n = st.slider("Batas jumlah baris untuk diproses (hemat waktu)", 50, max(2000, len(df)), min(len(df), 1000), step=50)

# Terapkan filter tanggal (sebelum analisis)
filtered_df = df.copy()
if f_start and f_end:
    ser = pd.to_datetime(filtered_df["Tanggal"]).dt.date
    mask = (ser >= f_start) & (ser <= f_end)
    filtered_df = filtered_df[mask]

filtered_df = filtered_df.head(sample_n).reset_index(drop=True)

# -----------------------------
# Analisis: preprocessing + model
# -----------------------------
with st.spinner("🔄 Preprocessing & klasifikasi dengan IndoBERT..."):
    cleaned = filtered_df["Komentar"].astype(str).apply(clean_text)
    labels, confs = predict_batch(cleaned.tolist(), batch_size=32)

    result = filtered_df.assign(
        Komentar_Bersih = cleaned,
        Sentimen = labels,
        Kepercayaan = np.round(confs, 4)
    )

# -----------------------------
# Kartu metrik ringkas
# -----------------------------
st.markdown('<div class="fade-in">', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
total = len(result)
pos = (result["Sentimen"] == "positif").sum()
neu = (result["Sentimen"] == "netral").sum()
neg = (result["Sentimen"] == "negatif").sum()
avg_conf = float(result["Kepercayaan"].mean()) if total else np.nan

for title, value in [
    ("Total Komentar", f"{total:,}"),
    ("Positif", f"{pos:,}"),
    ("Netral", f"{neu:,}"),
    ("Negatif", f"{neg:,}"),
]:
    with [m1, m2, m3, m4][["Total Komentar","Positif","Netral","Negatif"].index(title)]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Insight singkat otomatis
with st.expander("💡 Insight Otomatis"):
    dom = max([("positif", pos), ("netral", neu), ("negatif", neg)], key=lambda x: x[1])[0] if total else "-"
    st.write(
        f"- **Dominan:** {dom.capitalize()} ({max(pos, neu, neg)} dari {total} komentar). "
        f"Rata-rata kepercayaan model: **{(avg_conf*100):.1f}%**."
        "\n- Gunakan filter tanggal di atas untuk melihat perubahan komposisi sentimen."
    )

# -----------------------------
# Navigasi halaman
# -----------------------------
def page_overview(df_out: pd.DataFrame):
    st.subheader("👀 Preview & Ringkasan Data")
    c1, c2 = st.columns([1,1])
    with c1:
        st.write("**5 Baris Pertama (setelah analisis)**")
        st.dataframe(df_out.head(5), use_container_width=True, height=260)
    with c2:
        st.write("**Statistik Deskriptif (panjang teks & kepercayaan)**")
        stats = pd.DataFrame({
            "panjang_komentar": df_out["Komentar_Bersih"].str.split().map(len),
            "kepercayaan": df_out["Kepercayaan"],
        }).describe().T
        st.dataframe(stats, use_container_width=True, height=260)

    # Download seluruh hasil analisis yang sedang ditampilkan
    st.markdown("### ⬇️ Unduh Hasil Analisis")
    buff = io.StringIO()
    df_out.to_csv(buff, index=False)
    st.download_button("Download CSV (hasil analisis)", data=buff.getvalue(), file_name="hasil_analisis_sentimen.csv", mime="text/csv")


def page_analisis(df_out: pd.DataFrame):
    st.subheader("📈 Distribusi & Word Cloud")

    # Distribusi (pie)
    dist = df_out["Sentimen"].value_counts().reset_index()
    dist.columns = ["Sentimen", "Jumlah"]
    fig_pie = px.pie(
        dist, names="Sentimen", values="Jumlah",
        title="Distribusi Sentimen",
        color="Sentimen",
        color_discrete_map={"positif":"#2ECC71","netral":ACCENT,"negatif":"#E74C3C"},
        hole=.35
    )
    fig_pie.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=420)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Bar
    fig_bar = px.bar(
        dist.sort_values("Jumlah", ascending=False),
        x="Sentimen", y="Jumlah", text_auto=True,
        color="Sentimen",
        color_discrete_map={"positif":"#2ECC71","netral":ACCENT,"negatif":"#E74C3C"},
        title="Jumlah Komentar per Sentimen", template="simple_white"
    )
    fig_bar.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=420)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Word cloud per label
    st.markdown("#### ☁️ Word Cloud per Sentimen")
    c1, c2, c3 = st.columns(3)
    for label, col in zip(["positif","netral","negatif"], [c1,c2,c3]):
        text = " ".join(df_out.loc[df_out["Sentimen"]==label, "Komentar_Bersih"].astype(str))
        if len(text.strip()) == 0:
            col.info(f"Tidak ada teks untuk **{label}**.")
            continue
        wc = WordCloud(width=700, height=380, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(6,3.2))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        col.pyplot(fig, use_container_width=True)


def page_visual(df_out: pd.DataFrame):
    st.subheader("📊 Visualisasi Waktu & Kepercayaan")

    # Line (trend harian)
    tmp = df_out.copy()
    tmp["Tanggal"] = pd.to_datetime(tmp["Tanggal"])
    trend = tmp.groupby([tmp["Tanggal"].dt.to_period("D"), "Sentimen"]).size().reset_index(name="Jumlah")
    trend["Tanggal"] = trend["Tanggal"].astype(str)

    fig_line = px.line(
        trend, x="Tanggal", y="Jumlah", color="Sentimen", markers=True,
        color_discrete_map={"positif":"#2ECC71","netral":ACCENT,"negatif":"#E74C3C"},
        title="Trend Sentimen per Hari", template="simple_white"
    )
    fig_line.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=480)
    st.plotly_chart(fig_line, use_container_width=True)

    # Scatter (kepercayaan vs panjang komentar)
    df_out = df_out.copy()
    df_out["Panjang"] = df_out["Komentar_Bersih"].str.split().map(len)
    fig_scatter = px.scatter(
        df_out, x="Panjang", y="Kepercayaan", color="Sentimen",
        color_discrete_map={"positif":"#2ECC71","netral":ACCENT,"negatif":"#E74C3C"},
        hover_data=["No","Tanggal","Komentar"],
        title="Kepercayaan Model vs Panjang Komentar", template="simple_white",
        trendline="ols" if len(df_out) >= 10 else None
    )
    fig_scatter.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=480)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Tabel + download hasil filter (opsional filter sentimen lokal)
    with st.expander("📋 Tabel & Unduh (berdasarkan filter sentimen)"):
        opts = st.multiselect("Pilih label sentimen", ["positif","netral","negatif"], default=["positif","netral","negatif"])
        tab = df_out[df_out["Sentimen"].isin(opts)]
        st.dataframe(tab, use_container_width=True, height=360)

        buff = io.StringIO()
        tab.to_csv(buff, index=False)
        st.download_button("💾 Download CSV (ter-filter)", data=buff.getvalue(), file_name="hasil_filter_sentimen.csv", mime="text/csv")

# -----------------------------
# Router halaman
# -----------------------------
if page == "Overview":
    page_overview(result)
elif page == "Analisis Sentimen":
    page_analisis(result)
else:
    page_visual(result)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 14px; opacity: .7; font-size: 12px;">
      Dibuat dengan ❤️ — IndoBERT + Streamlit • Tema #2E86C1 / #F4D03F
    </div>
    """,
    unsafe_allow_html=True,
)
