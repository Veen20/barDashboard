import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# ==============================
# Konfigurasi Halaman Streamlit
# ==============================
st.set_page_config(page_title="transaksi_komentar", layout="wide")
st.title("📊 Dashboard Transaksi & Sentimen eSIGNAL")

# ==============================
# Autentikasi ke Google Sheets
# ==============================
try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_SHEET_CREDENTIALS"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
except Exception as e:
    st.error("❌ Gagal autentikasi dengan Google Sheets.")
    st.stop()

# ==============================
# Akses Spreadsheet & Worksheet
# ==============================
try:
    # Pastikan nama file spreadsheet sama persis di Google Drive
    spreadsheet = client.open("transaksi_komentar")
    sheet_transaksi = spreadsheet.worksheet("transaksi")
    sheet_komentar = spreadsheet.worksheet("komentar")
except Exception as e:
    st.error("❌ Gagal membuka spreadsheet atau worksheet. Periksa nama dan aksesnya.")
    st.stop()

# ==============================
# Convert ke DataFrame
# ==============================
df_transaksi = pd.DataFrame(sheet_transaksi.get_all_records())
df_komentar = pd.DataFrame(sheet_komentar.get_all_records())

# ==============================
# Tampilkan Data Transaksi
# ==============================
st.subheader("💳 Data Transaksi")
st.dataframe(df_transaksi)

if 'TANGGAL' in df_transaksi.columns:
    df_transaksi['TANGGAL'] = pd.to_datetime(df_transaksi['TANGGAL'], errors='coerce')
    angka_saja = df_transaksi.select_dtypes(include=['number'])
    if not angka_saja.empty:
        st.line_chart(df_transaksi.set_index('TANGGAL')[angka_saja.columns])
    else:
        st.info("Kolom numerik tidak ditemukan untuk visualisasi.")

# ==============================
# Tampilkan Data Komentar
# ==============================
st.subheader("💬 Data Komentar Publik")
st.dataframe(df_komentar)

if 'ulasan' in df_komentar.columns:
    st.markdown("#### Contoh Komentar:")
    for komentar in df_komentar['ulasan'].head(5):
        st.write(f"🗨️ {komentar}")
else:
    st.warning("Kolom 'ulasan' tidak ditemukan di sheet komentar.")
