import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Dashboard Interaktif e-SIGNAL", layout="wide")
st.title("📊 Dashboard Interaktif e-SIGNAL / Samsat")

# --- Load Data ---
df_transaksi = pd.read_excel("hasil_analisis_segmentasi.xlsx", parse_dates=['datetime'])
df_sentimen = pd.read_excel("hasil_analisis_sentimen_lengkap.xlsx", parse_dates=['timestamp'])

# Rename agar konsisten
df_transaksi.rename(columns={"datetime": "timestamp", "KODE DATI": "lokasi"}, inplace=True)

# --- Sidebar: Filter ---
st.sidebar.header("🔎 Filter Data")

lokasi_opsi = df_transaksi['lokasi'].dropna().unique()
lokasi_terpilih = st.sidebar.multiselect("Pilih Lokasi", options=lokasi_opsi, default=lokasi_opsi)

tanggal_min = df_transaksi['timestamp'].min().date()
tanggal_max = df_transaksi['timestamp'].max().date()
tanggal_range = st.sidebar.date_input("Pilih Rentang Tanggal", [tanggal_min, tanggal_max])

# --- Terapkan Filter ---
df_transaksi_filtered = df_transaksi[
    (df_transaksi['lokasi'].isin(lokasi_terpilih)) &
    (df_transaksi['timestamp'].dt.date >= tanggal_range[0]) &
    (df_transaksi['timestamp'].dt.date <= tanggal_range[1])
]

df_sentimen_filtered = df_sentimen[
    (df_sentimen['timestamp'].dt.date >= tanggal_range[0]) &
    (df_sentimen['timestamp'].dt.date <= tanggal_range[1])
]

# --- Statistik Ringkas ---
st.subheader("📈 Statistik Ringkas")
col1, col2 = st.columns(2)
col1.metric("Total Transaksi", f"{len(df_transaksi_filtered):,}")
col2.metric("Total Komentar", f"{len(df_sentimen_filtered):,}")

# --- Grafik Transaksi per Jam ---
st.subheader("⏰ Distribusi Transaksi per Jam")
df_transaksi_filtered['jam'] = df_transaksi_filtered['timestamp'].dt.hour
jam_count = df_transaksi_filtered['jam'].value_counts().sort_index()

fig1, ax1 = plt.subplots()
sns.barplot(x=jam_count.index, y=jam_count.values, ax=ax1)
ax1.set_xlabel("Jam")
ax1.set_ylabel("Jumlah Transaksi")
st.pyplot(fig1)

# --- Pie Chart Sentimen ---
st.subheader("🙂 Proporsi Sentimen Komentar")
sentimen_count = df_sentimen_filtered['kategori_sentimen'].value_counts()

fig2, ax2 = plt.subplots()
ax2.pie(sentimen_count, labels=sentimen_count.index, autopct='%1.1f%%', startangle=90)
st.pyplot(fig2)

# --- Word Cloud Komentar ---
st.subheader("🗣️ Word Cloud Komentar")
text = ' '.join(df_sentimen_filtered['komentar_bersih'].dropna())
if text:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis('off')
    st.pyplot(fig3)
else:
    st.warning("Tidak ada komentar pada filter saat ini.")