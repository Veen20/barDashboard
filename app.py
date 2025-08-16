import os
import time
from typing import List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import plotly.express as px

import gspread
from google.oauth2.service_account import Credentials

# ========= PAGE CONFIG =========
st.set_page_config(
    page_title="Sentimen e-SIGNAL – Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========= CUSTOM CSS =========
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
  --pos: #10b981;
  --neg: #ef4444;
  --neu: #f59e0b;
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
    predict_mode = st.radio(
        "Mode Prediksi",
        ["Hanya baris baru (default)", "Semua baris"],
        index=0
    )
    writeback = st.toggle("Tulis hasil kembali ke Google Sheets", value=True)

# ========= MODEL =========
@st.cache_resource(show_spinner=True)
def load_model() -> TextClassificationPipeline:
    model_name = "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
    return pipe

pipe = load_model()

# ========= GOOGLE SHEETS =========
@st.cache_resource
def connect_gspread():
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    client = gspread.authorize(creds)
    sh = client.open_by_url(st.secrets["spreadsheet"]["url"])
    return sh

sh = connect_gspread()

def load_df(sheet_name="Sheet1"):
    worksheet = sh.worksheet(sheet_name)
    df = pd.DataFrame(worksheet.get_all_records())
    if "ulasan" not in df.columns:
        st.warning("Kolom 'ulasan' tidak ditemukan di Google Sheets.")
        df["ulasan"] = ""
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now(tz="Asia/Jakarta")
    if "sentiment" not in df.columns:
        df["sentiment"] = np.nan
    if "score" not in df.columns:
        df["score"] = np.nan
    return df, worksheet

def update_sheet(df, worksheet):
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

df, worksheet = load_df()

# ========= PREDICT =========
LABEL_MAP = {"positive": "Positif", "negative": "Negatif", "neutral": "Netral"}

def predict_batch(texts: List[str]) -> List[Tuple[str, float]]:
    if not texts: return []
    outputs = pipe(texts)
    preds = []
    for scores in outputs:
        best = max(scores, key=lambda s: float(s["score"]))
        lab = LABEL_MAP.get(best["label"].lower(), best["label"])
        preds.append((lab, float(best["score"])))
    return preds

def need_prediction(row) -> bool:
    if predict_mode.startswith("Semua"):
        return True
    return pd.isna(row.get("sentiment")) or str(row.get("sentiment")).strip() == ""

mask = df.apply(need_prediction, axis=1) if len(df) else pd.Series([], dtype=bool)
to_pred = df.loc[mask, "ulasan"].fillna("").astype(str).tolist()

with st.spinner("Menghitung sentimen dengan IndoBERTweet..."):
    new_preds = predict_batch(to_pred)

if len(new_preds):
    df.loc[mask, "sentiment"] = [p[0] for p in new_preds]
    df.loc[mask, "score"] = [round(p[1], 4) for p in new_preds]
    if writeback:
        try:
            update_sheet(df, worksheet)
        except Exception as e:
            st.warning(f"Gagal menulis balik ke Google Sheets: {e}")

# ========= HEADER =========
st.markdown("""
<div class="deep-card">
  <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
    <div class="badge">IndoBERTweet</div>
    <div class="badge">Realtime Google Sheets</div>
    <div class="badge">Dark • Deep</div>
  </div>
  <div class="sep"></div>
  <h1 style="margin:0">Dashboard Sentimen e-SIGNAL</h1>
  <p style="color:var(--muted); margin:.2rem 0 0;">Pantau persepsi publik terhadap layanan e-SIGNAL secara langsung.</p>
</div>
""", unsafe_allow_html=True)

# ========= KPI =========
pos = (df["sentiment"] == "Positif").sum()
neg = (df["sentiment"] == "Negatif").sum()
neu = (df["sentiment"] == "Netral").sum()
total = len(df)

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("📈 Positif", pos)
with c2: st.metric("📉 Negatif", neg)
with c3: st.metric("➖ Netral", neu)
with c4: st.metric("🧾 Total", total)

# ========= PIE =========
if total:
    fig = px.pie(df, names="sentiment", title="Distribusi Sentimen", hole=0.45)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#e6ecff"
    )
    st.plotly_chart(fig, use_container_width=True)

# ========= TREND =========
if "timestamp" in df.columns and total:
    tmp = df.copy()
    tmp["dt"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
    tmp = tmp.dropna(subset=["dt"])
    if len(tmp):
        daily = tmp.groupby([pd.Grouper(key="dt", freq="D"), "sentiment"]).size().reset_index(name="jumlah")
        fig2 = px.line(daily, x="dt", y="jumlah", color="sentiment", title="Tren Harian per Sentimen")
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e6ecff")
        st.plotly_chart(fig2, use_container_width=True)

# ========= TABLE =========
st.markdown("#### 🔎 Sampel Ulasan Terbaru")
if total:
    show_cols = [c for c in ["timestamp", "ulasan", "sentiment", "score"] if c in df.columns]
    st.dataframe(df[show_cols].sort_values(by="timestamp", ascending=False).head(100), use_container_width=True)
else:
    st.info("Belum ada data di Google Sheets. Tambahkan kolom **ulasan** dan (opsional) **timestamp**.")
