import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Transaksi & Sentimen eSIGNAL", layout="wide")
st.title("📊 Dashboard Transaksi & Sentimen eSIGNAL")

# Autentikasi Google Sheets
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

try:
    creds_dict = json.loads(st.secrets["GOOGLE_SHEET_CREDENTIALS"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)

    # Gunakan open_by_url untuk akurasi
    spreadsheet = client.open("transaksi_komentar") 

    # Ambil worksheet
    sheet_transaksi = spreadsheet.worksheet("transaksi")
    sheet_komentar = spreadsheet.worksheet("komentar")

    # Konversi ke DataFrame
    df_transaksi = pd.DataFrame(sheet_transaksi.get_all_records())
    df_komentar = pd.DataFrame(sheet_komentar.get_all_records())

    # ======================
    # TAMPILKAN DATA TRANSAKSI
    # ======================
    st.subheader("💳 Data Transaksi")
    st.dataframe(df_transaksi)

    # Visualisasi jika ada kolom TANGGAL
    if 'TANGGAL' in df_transaksi.columns:
        df_transaksi['TANGGAL'] = pd.to_datetime(df_transaksi['TANGGAL'], errors='coerce')
        st.line_chart(df_transaksi.set_index('TANGGAL').select_dtypes(include=['number']))

    # ======================
    # TAMPILKAN DATA KOMENTAR
    # ======================
    st.subheader("💬 Data Komentar Publik")
    st.dataframe(df_komentar)

    # Tampilkan beberapa komentar
    if 'ulasan' in df_komentar.columns:
        st.markdown("#### Contoh Komentar:")
        for komentar in df_komentar['ulasan'].head(5):
            st.write(f"🗨️ {komentar}")

except Exception as e:
    st.error("❌ Gagal membuka spreadsheet atau worksheet. Periksa nama dan aksesnya.")
    st.exception(e)


