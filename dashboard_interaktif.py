import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 🔹 Konfigurasi Halaman
# ------------------------------
st.set_page_config(page_title="Dashboard Transaksi & Sentimen eSIGNAL", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Dashboard Transaksi & Sentimen eSIGNAL")

# ------------------------------
# 🔹 Autentikasi Google Sheets
# ------------------------------
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

    # ------------------------------
    # 🔹 Tab Layout
    # ------------------------------
    tab1, tab2, tab3 = st.tabs(["📄 Data Transaksi", "💬 Komentar Publik", "📈 Visualisasi & Clustering"])

    # ------------------------------
    # 🔹 Tab 1: Data Transaksi
    # ------------------------------
    with tab1:
        st.subheader("💳 Tabel Data Transaksi")
        st.dataframe(df_transaksi, use_container_width=True)

        if 'TANGGAL' in df_transaksi.columns:
            df_transaksi['TANGGAL'] = pd.to_datetime(df_transaksi['TANGGAL'], errors='coerce')
            st.markdown("### 📆 Visualisasi Tren Transaksi")
            st.line_chart(df_transaksi.set_index('TANGGAL').select_dtypes(include=['number']))

    # ------------------------------
    # 🔹 Tab 2: Komentar Publik
    # ------------------------------
    with tab2:
        st.subheader("💬 Tabel Komentar Publik")
        st.dataframe(df_komentar, use_container_width=True)

        if 'ulasan' in df_komentar.columns:
            st.markdown("### 🔍 Beberapa Komentar Terbaru")
            for i, row in df_komentar.head(5).iterrows():
                with st.expander(f"🗨️ Komentar {i+1}"):
                    st.write(row['ulasan'])

    # ------------------------------
    # 🔹 Tab 3: Visualisasi Clustering
    # ------------------------------
    with tab3:
        st.subheader("🔎 Visualisasi Clustering Transaksi")

        if 'cluster_kmeans' in df_transaksi.columns and 'jam_only' in df_transaksi.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=df_transaksi, x='jam_only', y='cluster_kmeans', hue='cluster_kmeans', palette='tab10', s=100, ax=ax)
            ax.set_title("Distribusi Cluster Berdasarkan Jam Transaksi")
            st.pyplot(fig)
        else:
            st.info("⚠️ Kolom 'cluster_kmeans' dan 'jam_only' belum tersedia di data transaksi.")

except Exception as e:
    st.error("❌ Gagal membuka spreadsheet atau worksheet. Periksa nama dan aksesnya.")
    st.exception(e)
