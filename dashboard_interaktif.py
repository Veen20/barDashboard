import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# =====================
# Konfigurasi Awal
# =====================
st.set_page_config(page_title="Dashboard e-SIGNAL", layout="wide")
st.title("📊 Dashboard Interaktif e-SIGNAL / SAMSAT")
st.caption("Visualisasi interaktif data transaksi dan komentar masyarakat untuk mendukung perbaikan layanan publik.")

# =====================
# Load Data
# =====================
def load_data():
    df_transaksi = pd.read_excel("hasil_analisis_segmentasi.xlsx")
    df_transaksi['timestamp'] = pd.to_datetime(df_transaksi['datetime'])  # Ganti dari datetime

    df_sentimen = pd.read_excel("hasil_analisis_sentimen_lengkap.xlsx", parse_dates=["timestamp"])
    df_sentimen.rename(columns={"label_sentimen": "sentimen"}, inplace=True)  # Standarisasi nama kolom sentimen

    return df_transaksi, df_sentimen


# =====================
# Sidebar Filter
# =====================
st.sidebar.header("📅 Filter Tanggal")
tanggal_mulai = st.sidebar.date_input("Tanggal Mulai", df_transaksi['timestamp'].min().date())
tanggal_akhir = st.sidebar.date_input("Tanggal Akhir", df_transaksi['timestamp'].max().date())

mask_transaksi = (df_transaksi['timestamp'].dt.date >= tanggal_mulai) & (df_transaksi['timestamp'].dt.date <= tanggal_akhir)
df_transaksi_filtered = df_transaksi[mask_transaksi]

mask_sentimen = (df_sentimen['timestamp'].dt.date >= tanggal_mulai) & (df_sentimen['timestamp'].dt.date <= tanggal_akhir)
df_sentimen_filtered = df_sentimen[mask_sentimen]

# =====================
# Bar Jam Transaksi
# =====================
st.subheader("⏰ Distribusi Jam Transaksi")
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.countplot(data=df_transaksi_filtered, x='jam', order=sorted(df_transaksi['jam'].unique()), palette='viridis')
plt.xticks(rotation=45)
ax1.set_title("Jumlah Transaksi per Jam")
st.pyplot(fig1)

# =====================
# Linechart Volume Transaksi per Hari
# =====================
st.subheader("📈 Tren Volume Transaksi Harian")
df_transaksi_filtered['tanggal'] = df_transaksi_filtered['timestamp'].dt.date
harian = df_transaksi_filtered.groupby('tanggal').size()
st.line_chart(harian)

# =====================
# WordCloud Positif & Negatif
# =====================
st.subheader("💬 WordCloud Komentar Positif dan Negatif")
col1, col2 = st.columns(2)

with col1:
    positif = df_sentimen_filtered[df_sentimen_filtered['sentimen'] == 'positif']['komentar_bersih'].str.cat(sep=' ')
    st.markdown("**Komentar Positif**")
    if positif.strip():
        wc_pos = WordCloud(width=400, height=300, background_color='white').generate(positif)
        fig2, ax2 = plt.subplots()
        ax2.imshow(wc_pos, interpolation='bilinear')
        ax2.axis('off')
        st.pyplot(fig2)
    else:
        st.info("Tidak ada komentar positif.")

with col2:
    negatif = df_sentimen_filtered[df_sentimen_filtered['sentimen'] == 'negatif']['komentar_bersih'].str.cat(sep=' ')
    st.markdown("**Komentar Negatif**")
    if negatif.strip():
        wc_neg = WordCloud(width=400, height=300, background_color='white').generate(negatif)
        fig3, ax3 = plt.subplots()
        ax3.imshow(wc_neg, interpolation='bilinear')
        ax3.axis('off')
        st.pyplot(fig3)
    else:
        st.info("Tidak ada komentar negatif.")

# =====================
# Distribusi Sentimen
# =====================
st.subheader("📊 Distribusi Sentimen Komentar")
fig4, ax4 = plt.subplots()
sns.countplot(data=df_sentimen_filtered, x='sentimen', palette='Set2')
ax4.set_title("Jumlah Komentar per Sentimen")
st.pyplot(fig4)

# =====================
# Tren Sentimen Harian
# =====================
st.subheader("📉 Tren Sentimen Harian")
df_sentimen_filtered['tanggal'] = df_sentimen_filtered['timestamp'].dt.date
tren_sentimen = df_sentimen_filtered.groupby(['tanggal', 'sentimen']).size().unstack(fill_value=0)
if not tren_sentimen.empty:
    st.line_chart(tren_sentimen)
else:
    st.warning("Data tren sentimen tidak tersedia di rentang tanggal ini.")

# =====================
# Tabel Komentar Negatif
# =====================
with st.expander("🗒️ Daftar Komentar Negatif (untuk evaluasi layanan)"):
    komentar_negatif = df_sentimen_filtered[df_sentimen_filtered['sentimen'] == 'negatif']
    st.dataframe(komentar_negatif[['timestamp', 'komentar_bersih']].sort_values(by='timestamp', ascending=False))

# =====================
# Cluster Jam (jika ada kolom 'cluster')
# =====================
if 'cluster' in df_transaksi_filtered.columns:
    st.subheader("🧩 Segmentasi Waktu (Hasil Clustering)")
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df_transaksi_filtered, x='cluster', order=sorted(df_transaksi_filtered['cluster'].unique()), palette='coolwarm')
    ax5.set_title("Jumlah Transaksi per Cluster Waktu")
    st.pyplot(fig5)

