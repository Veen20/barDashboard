import streamlit as st
import pandas as pd
import gspread
import matplotlib.pyplot as plt
import seaborn as sns
from oauth2client.service_account import ServiceAccountCredentials
import json
from io import BytesIO

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
# TABS UNTUK NAVIGASI
# ===============================
tab1, tab2, tab3 = st.tabs(["📑 Data Mentah", "📈 Visualisasi", "📬 Komentar Publik"])

# ===============================
# TAB 1: TAMPILKAN DATA MENTAH
# ===============================
with tab1:
    st.subheader("💳 Data Transaksi")
    st.dataframe(df_transaksi, use_container_width=True)

    st.subheader("💬 Data Komentar")
    st.dataframe(df_komentar, use_container_width=True)

# ===============================
# TAB 2: VISUALISASI
# ===============================
with tab2:
    st.subheader("📊 Visualisasi Data")

    col1, col2 = st.columns(2)

    # Distribusi Jam Transaksi
    with col1:
        if 'jam_only' in df_transaksi.columns:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            sns.histplot(df_transaksi['jam_only'], bins=24, kde=True, color='skyblue', edgecolor='black', ax=ax1)
            ax1.set_title("Distribusi Jam Transaksi")
            ax1.set_xlabel("Jam (0-23)")
            ax1.set_ylabel("Frekuensi")
            st.pyplot(fig1)

    # Distribusi Sentimen
    with col2:
        if 'kategori_sentimen' in df_komentar.columns:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            df_komentar['kategori_sentimen'].value_counts().plot(kind='bar', color=['red', 'green', 'blue'], ax=ax2)
            ax2.set_title("Distribusi Sentimen Pengguna SIGNAL")
            ax2.set_xlabel("Kategori Sentimen")
            ax2.set_ylabel("Jumlah Komentar")
            st.pyplot(fig2)

    # Distribusi Sentimen per Hari
    if 'hari' in df_komentar.columns and 'kategori_sentimen' in df_komentar.columns:
        st.markdown("### 📆 Distribusi Sentimen per Hari")
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        sns.histplot(data=df_komentar, x="hari", hue="kategori_sentimen", multiple="stack", ax=ax3)
        ax3.set_ylabel("Jumlah Komentar")
        st.pyplot(fig3)

    # Visualisasi Clustering Hari (jika ada)
    if 'kategori_kmeans' in df_transaksi.columns and 'hari' in df_transaksi.columns:
        st.markdown("### 📌 Distribusi Profil Hari (Clustering)")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        pd.crosstab(df_transaksi['hari'], df_transaksi['kategori_kmeans']).plot(kind='bar', ax=ax4)
        ax4.set_title("Distribusi Profil Hari")
        ax4.set_xlabel("Hari")
        ax4.set_ylabel("Frekuensi")
        st.pyplot(fig4)

# ===============================
# TAB 3: ULASAN
# ===============================
with tab3:
    st.subheader("🗨️ Contoh Komentar Pengguna")
    if 'ulasan' in df_komentar.columns:
        for i, komentar in enumerate(df_komentar['ulasan'].head(10), start=1):
            st.write(f"**{i}.** {komentar}")

# ===============================
# CATATAN AKHIR
# ===============================
st.markdown("""
---
📌 *Dashboard ini menampilkan informasi transaksi dan persepsi publik terhadap layanan pembayaran PKB melalui aplikasi SIGNAL. Data diperoleh dari Google Spreadsheet dan diperbarui secara langsung.*
""")
