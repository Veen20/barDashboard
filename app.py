import os
import time
from typing import List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from streamlit_gsheets import GSheetsConnection
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# ========= THEME & PAGE =========
st.set_page_config(
    page_title="Sentimen e-SIGNAL – Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Deep modern look
st.markdown("""
<style>
:root {
  --bg-deep: #0b1020;
  --panel: #111a2b;
  --panel-2: #0e1524;
  --accent: #5b8cff;
  --accent-2: #8c5bff;
  --text: #e6ecff;
  --muted: #94a3b8;
  --pos: #10b981;   /* green */
  --neg: #ef4444;   /* red */
  --neu: #f59e0b;   /* amber */
}
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 800px at 10% 10%, #0f1b35 0%, var(--bg-deep) 45%, #070b15 100%) !important;
  color: var(--text);
}
[data-testid="stHeader"] { background: linear-gradient(180deg, rgba(7,11,21,.9), rgba(7,11,21,0)); }
.block-container { padding-top: 1rem; }
.deep-card {
  background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
  border: 1px solid rgba(255,255,255,0.04);
  border-radius: 18px; padding: 18px; box-shadow: 0 10px 30px rgba(0,0,0,.35);
}
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px; font-size:.78rem; font-weight:600;
  letter-spacing:.3px; border:1px solid rgba(255,255,255,.08); background:rgba(255,255,255,.04);
}
.kpi { font-size: 28px; font-weight: 700; letter-spacing: .3px; }
.sep { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,.08), transparent); margin: 12px 0; }
a { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ========= SIDEBAR =========
with st.sidebar:
    st.markdown("### 💬 Sentimen e-SIGNAL")
    st.caption("Dashboard modern • IndoBERTweet • Google Sheets realtime")
    refresh_sec = st.slider("Auto-refresh (detik)", 10, 120, 30, help="Perbarui data secara periodik")
    predict_mode = st.radio(
        "Mode Prediksi",
        ["Hanya baris baru (disarankan)", "Semua baris (paksa ulang)"],
        index=0,
        help="‘Semua baris’ akan menghitung ulang seluruh kolom, gunakan bila kamu mengganti model."
    )
    writeback = st.toggle("Tulis hasil ke Google Sheets", value=True,
                          help="Jika aktif, kolom sentimen & skor akan diperbarui di sheet.")

# Auto refresh
st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh_key")

# ========= MODEL =========
@st.cache_resource(show_spinner=True)
def load_model() -> TextClassificationPipeline:
    model_name = "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
    return pipe

pipe = load_model()

# ========= SHEETS CONNECTION =========
# Secrets required:   [connections.gsheets]   spreadsheet = "https://docs.google.com/..."
@st.cache_resource(show_spinner=False)
def connect_gsheets() -> GSheetsConnection:
    return st.connection("gsheets", type=GSheetsConnection)

conn = connect_gsheets()

@st.cache_data(ttl=15, show_spinner=False)
def load_df(sheet: str = "Sheet1") -> pd.DataFrame:
    df = conn.read(worksheet=sheet, ttl=0)  # always fetch fresh on call; caching at Streamlit level
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["timestamp", "ulasan"])
    # Normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    # Ensure required columns
    if "ulasan" not in [c.lower() for c in df.columns]:
        # Try alternate names
        text_col = None
        for c in df.columns:
            if c.lower() in ("review", "komentar", "text", "pesan"):
                text_col = c
                break
        if text_col:
            df = df.rename(columns={text_col: "ulasan"})
        else:
            df["ulasan"] = ""
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now(tz="Asia/Jakarta")
    # Ensure sentiment columns
    if "sentiment" not in df.columns:
        df["sentiment"] = np.nan
    if "score" not in df.columns:
        df["score"] = np.nan
    return df

df = load_df()

# ========= PREDICT =========
LABEL_MAP = {
    "positive": "Positif",
    "negative": "Negatif",
    "neutral":  "Netral"
}

def predict_batch(texts: List[str]) -> List[Tuple[str, float]]:
    if not texts:
        return []
    preds = []
    # pipe returns list of list of dicts [{label, score}, ...]
    outputs = pipe(texts)
    for scores in outputs:
        # pick max
        best = max(scores, key=lambda s: float(s["score"]))
        lab = LABEL_MAP.get(best["label"].lower(), best["label"])
        preds.append((lab, float(best["score"])))
    return preds

def need_prediction(row) -> bool:
    if predict_mode.startswith("Semua"):
        return True
    return (pd.isna(row.get("sentiment")) or str(row.get("sentiment")).strip() == "")

mask = df.apply(need_prediction, axis=1) if len(df) else pd.Series([], dtype=bool)
to_pred = df.loc[mask, "ulasan"].fillna("").astype(str).tolist()
with st.spinner("Menghitung sentimen dengan IndoBERTweet..."):
    new_preds = predict_batch(to_pred)

if len(new_preds):
    df.loc[mask, "sentiment"] = [p[0] for p in new_preds]
    df.loc[mask, "score"] = [round(p[1], 4) for p in new_preds]
    # Optionally write-back
    if writeback:
        try:
            conn.update(data=df)
        except Exception as e:
            st.warning(f"Gagal menulis balik ke Google Sheets: {e}")

# ========= HEADER =========
st.markdown(
    """
<div class="deep-card">
  <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
    <div class="badge">IndoBERTweet • 3 kelas</div>
    <div class="badge">Realtime Google Sheets</div>
    <div class="badge">Dark • Deep</div>
  </div>
  <div class="sep"></div>
  <h1 style="margin:0">Dashboard Sentimen e-SIGNAL</h1>
  <p style="color:var(--muted); margin:.2rem 0 0;">Pantau persepsi publik terhadap layanan e-SIGNAL secara langsung.</p>
</div>
""",
    unsafe_allow_html=True
)

# ========= KPIs =========
pos = (df["sentiment"] == "Positif").sum() if "sentiment" in df else 0
neg = (df["sentiment"] == "Negatif").sum() if "sentiment" in df else 0
neu = (df["sentiment"] == "Netral").sum()  if "sentiment" in df else 0
total = len(df)

c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    st.markdown('<div class="deep-card"><div class="kpi">📈 Positif</div><div style="color:var(--pos); font-size:22px; font-weight:700;">{}</div></div>'.format(pos), unsafe_allow_html=True)
with c2:
    st.markdown('<div class="deep-card"><div class="kpi">📉 Negatif</div><div style="color:var(--neg); font-size:22px; font-weight:700;">{}</div></div>'.format(neg), unsafe_allow_html=True)
with c3:
    st.markdown('<div class="deep-card"><div class="kpi">➖ Netral</div><div style="color:var(--neu); font-size:22px; font-weight:700;">{}</div></div>'.format(neu), unsafe_allow_html=True)
with c4:
    st.markdown('<div class="deep-card"><div class="kpi">🧾 Total</div><div style="font-size:22px; font-weight:700;">{}</div></div>'.format(total), unsafe_allow_html=True)

st.write(" ")

# ========= DISTRIBUSI =========
if total:
    fig = px.pie(
        df.dropna(subset=["sentiment"]),
        names="sentiment",
        title="Distribusi Sentimen",
        hole=0.45
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color="#e6ecff", title_font_size=18,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

# ========= TREN WAKTU =========
if total and "timestamp" in df.columns:
    tmp = df.copy()
    tmp["dt"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
    tmp = tmp.dropna(subset=["dt"])
    if len(tmp):
        daily = (tmp.groupby([pd.Grouper(key="dt", freq="D"), "sentiment"])
                 .size().reset_index(name="jumlah"))
        fig2 = px.line(daily, x="dt", y="jumlah", color="sentiment", title="Tren Harian per Sentimen")
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color="#e6ecff", title_font_size=18
        )
        st.plotly_chart(fig2, use_container_width=True)

# ========= TABEL =========
st.markdown("#### 🔎 Sampel Ulasan Terbaru")
if total:
    show_cols = [c for c in ["timestamp", "ulasan", "sentiment", "score"] if c in df.columns]
    st.dataframe(df[show_cols].sort_values(by=show_cols[0], ascending=False).head(100), use_container_width=True)
else:
    st.info("Belum ada data di Google Sheets. Tambahkan kolom **ulasan** dan (opsional) **timestamp**.")

st.markdown(
    """
<div class="deep-card" style="margin-top:12px;">
  <b>Format kolom yang disarankan (Google Sheets):</b>
  <ul>
    <li><code>timestamp</code> (datetime, opsional)</li>
    <li><code>ulasan</code> (teks)</li>
    <li><code>sentiment</code> (akan diisi otomatis)</li>
    <li><code>score</code> (akan diisi otomatis)</li>
  </ul>
</div>
""",
    unsafe_allow_html=True
)
