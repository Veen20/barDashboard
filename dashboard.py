import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

st.set_page_config(page_title="Dashboard eSIGNAL", layout="wide")
st.title("📊 Dashboard Transaksi & Sentimen eSIGNAL")

# Autentikasi Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(st.secrets["GOOGLE_SHEET_CREDENTIALS"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

# 🔹 Buka Spreadsheet utama
spreadsheet = client.open("spreadsheet Analisis Transaksi & komentar")

# Ambil masing-masing worksheet
sheet_transaksi = spreadsheet.worksheet("Transaksi")
sheet_komentar = spreadsheet.worksheet("Komentar")

# Ubah ke DataFrame
df_transaksi = pd.DataFrame(sheet_transaksi.get_all_records())
df_komentar = pd.DataFrame(sheet_komentar.get_all_records())

# ==============================
# TAMPILKAN DATA TRANSAKSI
# ==============================
st.subheader("💳 Data Transaksi")
st.dataframe(df_transaksi)

# Contoh visualisasi transaksi (jika ada kolom waktu)
if 'tanggal' in df_transaksi.columns:
    df_transaksi['tanggal'] = pd.to_datetime(df_transaksi['tanggal'], errors='coerce')
    st.line_chart(df_transaksi.set_index('tanggal').select_dtypes(include=['number']))

# ==============================
# TAMPILKAN DATA KOMENTAR
# ==============================
st.subheader("💬 Data Komentar Publik")
st.dataframe(df_komentar)

# Tampilkan beberapa ulasan
if 'ulasan' in df_komentar.columns:
    st.markdown("#### Contoh Komentar:")
    for komentar in df_komentar['ulasan'].head(5):
        st.write(f"🗨️ {komentar}")
