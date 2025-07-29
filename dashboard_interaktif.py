import streamlit as st
import pandas as pd
import gspread
import matplotlib.pyplot as plt
import seaborn as sns
from oauth2client.service_account import ServiceAccountCredentials
import json

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Dashboard Transaksi & Sentimen eSIGNAL",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("## 📊 Dashboard Transaksi & Sentimen eSIGNAL")

# ===============================
# AUTENTIKASI DAN AMBIL DATA
# ===============================
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

try:
    creds_dict = json.loads(st.secrets["GOOGLE_SHEET_CREDENTIALS"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)

    spreadsheet = client.open("transaksi_komentar")
    sheet_transaksi = spreadsheet.worksheet("transaksi")
    sheet_komentar = spreadsheet.worksheet("komentar")

    df_transaksi = pd.DataFrame(sheet_transaksi.get_all_records())
    df_komentar = pd.DataFrame(sheet_komentar.get_all_records())

except Exception as e:
    st.error("❌ Gagal membuka spreadsheet atau worksheet. Periksa nama dan aksesnya.")
    st.exception(e)
    st.stop()

# ===============================
# TABS UTAMA
# ===============================
tab_data, tab_visual_transaksi, tab_visual_komentar, tab_komentar = st.tabs(
    ["📑 Data Mentah", "📈 Visualisasi Transaksi", "💬 Visualisasi Komentar", "🗨️ Komentar Terbaru"]
)

# ===============================
# TAB 1: DATA MENTAH
# ===============================
with tab_data:
    st.subheader("💳 Data Transaksi")
    st.dataframe(df_transaksi, use_container_width=True)

    st.subheader("💬 Data Komentar")
    st.dataframe(df_komentar, use_container_width=True)

# ===============================
# TAB 2: VISUALISASI TRANSAKSI
# ===============================
with tab_visual_transaksi:
    st.subheader("📈 Visualisasi Data Transaksi")

    col1, col2 = st.columns(2)

    # Visualisasi Distribusi Jam Transaksi
    with col1:
        if 'jam_only' in df_transaksi.columns:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            sns.histplot(df_transaksi['jam_only'], bins=24, kde=True, color='skyblue', edgecolor='black', ax=ax1)
            ax1.set_title("Distribusi Jam Transaksi")
            ax1.set_xlabel("Jam (0-23)")
            ax1.set_ylabel("Frekuensi")
            st.pyplot(fig1)

    # Visualisasi Clustering Hari (kalau ada)
    with col2:
        if 'kategori_kmeans' in df_transaksi.columns and 'hari' in df_transaksi.columns:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            pd.crosstab(df_transaksi['hari'], df_transaksi['kategori_kmeans']).plot(kind='bar', ax=ax2)
            ax2.set_title("Distribusi Profil Hari Berdasarkan Clustering")
            ax2.set_xlabel("Hari")
            ax2.set_ylabel("Frekuensi")
            st.pyplot(fig2)

# ===============================
# TAB 3: VISUALISASI KOMENTAR
# ===============================
with tab_visual_komentar:
    st.subheader("💬 Visualisasi Data Komentar Publik")

    col3, col4 = st.columns(2)

    # Distribusi Sentimen
    with col3:
        if 'kategori_sentimen' in df_komentar.columns:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            df_komentar['kategori_sentimen'].value_counts().plot(kind='bar', color=['red', 'green', 'blue'], ax=ax3)
            ax3.set_title("Distribusi Sentimen Pengguna SIGNAL")
            ax3.set_xlabel("Kategori Sentimen")
            ax3.set_ylabel("Jumlah Komentar")
            st.pyplot(fig3)

    # Distribusi Sentimen per Hari
    with col4:
        if 'hari' in df_komentar.columns and 'kategori_sentimen' in df_komentar.columns:
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            sns.histplot(data=df_komentar, x="hari", hue="kategori_sentimen", multiple="stack", ax=ax4)
            ax4.set_title("Distribusi Sentimen per Hari")
            ax4.set_xlabel("Hari")
            ax4.set_ylabel("Jumlah Komentar")
            st.pyplot(fig4)

# ===============================
# TAB 4: KOMENTAR TERBARU
# ===============================
with tab_komentar:
        st.dataframe(df_komentar, use_container_width=True)

        if 'ulasan' in df_komentar.columns:
            st.markdown("### 🔍 Beberapa Komentar Terbaru")
            for i, row in df_komentar.head(5).iterrows():
                with st.expander(f"🗨️ Komentar {i+1}"):
                    st.write(row['ulasan'])
# ===============================
# CATATAN PENUTUP
# ===============================
st.markdown("""
---
📌 *Dashboard ini menyajikan data transaksi dan persepsi publik terhadap layanan SIGNAL. Analisis dilakukan berdasarkan waktu transaksi, klaster hari, dan persepsi sentimen.*
""")
