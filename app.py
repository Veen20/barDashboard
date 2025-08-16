import streamlit as st
import pandas as pd
from transformers import pipeline

# === Koneksi ke Google Sheets ===
conn = st.connection("gsheets", type="gcsheets")

df = conn.read(worksheet="Sheet1")  # pastikan sheet namanya sesuai
df = df.dropna(subset=["ulasan"])   # hapus baris kosong

st.title("📊 Dashboard Analisis Sentimen IndoBERT + Google Sheets")

# === Load IndoBERT Sentiment Model ===
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="indobenchmark/indobert-base-p1")

sentiment_model = load_model()

# === Analisis Sentimen ===
if "sentimen" not in df.columns:
    df["sentimen"] = df["ulasan"].apply(lambda x: sentiment_model(str(x))[0]['label'])

# === Tampilkan Data ===
st.subheader("📋 Data Komentar + Hasil Sentimen")
st.dataframe(df)

# === Visualisasi ===
st.subheader("📊 Distribusi Sentimen")
st.bar_chart(df["sentimen"].value_counts())

# === Simpan ke Google Sheets lagi (optional) ===
if st.button("🔄 Update Sentimen ke GSheet"):
    conn.update(worksheet="Sheet1", data=df)
    st.success("Data sentimen berhasil diperbarui ke Google Sheet ✅")
