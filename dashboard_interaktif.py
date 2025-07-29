# Enhanced Streamlit Dashboard
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Layout config
st.set_page_config(layout="wide")
st.title("📊 Dashboard Transaksi & Sentimen eSIGNAL")

# Tabs
transaksi_tab, komentar_tab = st.tabs(["💳 Data Transaksi", "💬 Data Komentar"])

# ===========================
# Data Transaksi
# ===========================
with transaksi_tab:
    df_transaksi = st.session_state.get("df_transaksi")  # or load from Google Sheet if needed
    if df_transaksi is not None:
        df_transaksi['TANGGAL'] = pd.to_datetime(df_transaksi['TANGGAL'])

        st.subheader("📆 Tabel Transaksi Harian")
        st.dataframe(df_transaksi)

        # Total transaksi per hari
        st.subheader("📈 Total Transaksi per Hari")
        transaksi_harian = df_transaksi['TANGGAL'].dt.day_name().value_counts().reindex(
            ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
        ).fillna(0)
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.barplot(x=transaksi_harian.index, y=transaksi_harian.values, ax=ax1)
        ax1.set_ylabel("Jumlah Transaksi")
        ax1.set_xlabel("Hari")
        ax1.set_title("Distribusi Transaksi per Hari")
        st.pyplot(fig1)

        # Distribusi Jam Transaksi
        st.subheader("⏰ Distribusi Jam Transaksi")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.histplot(df_transaksi['jam_only'], bins=24, kde=True, ax=ax2, color='dodgerblue')
        ax2.set_xlabel("Jam (0-23)")
        ax2.set_ylabel("Frekuensi")
        ax2.set_title("Distribusi Jam Transaksi")
        st.pyplot(fig2)

        # Profil Hari (Jika sudah ada hasil clustering)
        if 'kategori_aturan' in df_transaksi.columns:
            st.subheader("📌 Distribusi Profil Jam Transaksi (Manual)")
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.countplot(data=df_transaksi, x='hari', hue='kategori_aturan', order=['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'])
            ax3.set_title("Distribusi Profil Hari Berdasarkan Kategori Waktu")
            st.pyplot(fig3)

# ===========================
# Data Komentar
# ===========================
with komentar_tab:
    df_komentar = st.session_state.get("df_komentar")
    if df_komentar is not None:
        st.subheader("📜 Tabel Komentar & Sentimen")
        st.dataframe(df_komentar)

        st.subheader("📌 Komentar Terbaru")
        for _, row in df_komentar.sort_values(by="tanggal", ascending=False).head(5).iterrows():
            st.write(f"🗨️ **{row['kategori_sentimen']}**: {row['komentar_bersih']}")

        # Distribusi sentimen
        st.subheader("📊 Distribusi Sentimen Pengguna")
        sentiment_count = df_komentar['kategori_sentimen'].value_counts()
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sentiment_count.plot(kind='bar', color=['red', 'blue', 'green'], ax=ax4)
        ax4.set_title("Distribusi Sentimen Pengguna Aplikasi SIGNAL")
        ax4.set_xlabel("Kategori Sentimen")
        ax4.set_ylabel("Jumlah Komentar")
        st.pyplot(fig4)

        # Distribusi Sentimen per Hari
        st.subheader("📆 Distribusi Sentimen per Hari")
        sentimen_hari = pd.crosstab(df_komentar['hari'], df_komentar['kategori_sentimen']).reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        sentimen_hari.plot(kind='bar', stacked=True, ax=ax5, colormap='Set2')
        ax5.set_title("Distribusi Sentimen Berdasarkan Hari")
        ax5.set_xlabel("Hari")
        ax5.set_ylabel("Jumlah Komentar")
        st.pyplot(fig5)
