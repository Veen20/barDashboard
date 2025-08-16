import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.util import ngrams
import re

# --- KONFIGURASI HALAMAN DAN GAYA (CSS) ---
st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk tema modern biru keunguan
st.markdown("""
<style>
/* Latar Belakang Utama */
.stApp {
    background-color: #0d1117;
    background-image: linear-gradient(160deg, #0d1117 0%, #21262d 100%);
}
/* Sidebar */
.css-1d391kg {
    background-color: rgba(25, 30, 40, 0.8);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}
/* Kartu Metrik */
.stMetric {
    background-color: rgba(40, 50, 70, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
/* Judul dan Teks */
h1, h2, h3 { color: #c9d1d9; }
/* Tombol */
.stButton>button {
    border: 2px solid #30363d; border-radius: 10px; color: #c9d1d9; background-color: #21262d;
}
.stButton>button:hover { border-color: #8b949e; color: #f0f6fc; }
</style>
""", unsafe_allow_html=True)

# --- PERBAIKAN BAGIAN NLTK ---
# Download stopwords jika belum ada (diperlukan untuk wordcloud)
try:
    nltk.data.find('corpora/stopwords')
except LookupError: # <-- PERBAIKAN DILAKUKAN DI SINI
    st.info("Mengunduh data NLTK (stopwords)... Ini hanya terjadi sekali.")
    nltk.download('stopwords')


# --- FUNGSI-FUNGSI UTAMA (KONEKSI, MODEL, ANALISIS) ---

@st.cache_resource
def connect_to_gsheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(creds)
    return client

@st.cache_data(ttl=300)
def fetch_data_from_gsheet(_gsheet_client, sheet_name, worksheet_name, comment_column):
    st.toast("🔄 Mengambil data baru dari Google Sheets...")
    try:
        sheet = _gsheet_client.open(sheet_name).worksheet(worksheet_name)
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Spreadsheet dengan nama '{sheet_name}' tidak ditemukan.")
        return pd.DataFrame()
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Worksheet dengan nama '{worksheet_name}' tidak ditemukan.")
        return pd.DataFrame()
        
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    proper_comment_column = None
    for col in df.columns:
        if col.lower() == comment_column.lower():
            proper_comment_column = col
            break
    
    if not proper_comment_column:
        st.error(f"Kolom '{comment_column}' tidak ditemukan. Kolom yang tersedia: {df.columns.tolist()}")
        return pd.DataFrame()
    
    df.rename(columns={proper_comment_column: 'komentar'}, inplace=True)

    if 'Tanggal' in df.columns:
        df['tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
        df.dropna(subset=['tanggal'], inplace=True)
    
    df = df[df['komentar'].astype(str).str.strip() != '']
    return df

@st.cache_resource
def load_sentiment_model():
    model = pipeline("sentiment-analysis", model="indobenchmark/indobert-base-p2-sentiment-classifier")
    return model

@st.cache_data
def analyze_sentiment(_df, _model):
    if 'komentar' not in _df.columns or _df.empty:
        return _df

    texts = _df['komentar'].tolist()
    results = _model(texts)
    
    _df['sentimen'] = [res['label'].replace('positive', 'Positif').replace('negative', 'Negatif').replace('neutral', 'Netral') for res in results]
    _df['skor'] = [res['score'] for res in results]
    return _df

def create_wordcloud(text_series, title):
    stopwords_indonesia = set(nltk.corpus.stopwords.words('indonesian'))
    text = ' '.join(text_series.astype(str).tolist())
    if not text: return
    
    wordcloud = WordCloud(width=800, height=400, background_color=None, colormap='viridis', stopwords=stopwords_indonesia, mode="RGBA").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_alpha(0)
    st.pyplot(fig)

# --- UI APLIKASI STREAMLIT ---

with st.sidebar:
    st.title("⚙️ Konfigurasi Dasbor")
    st.markdown("Masukkan detail Google Spreadsheet Anda di bawah ini.")
    
    NAMA_SPREADSHEET = st.text_input("Nama Google Spreadsheet", value="")
    NAMA_WORKSHEET = st.text_input("Nama Worksheet", value="Sheet1")
    KOLOM_KOMENTAR = st.text_input("Nama Kolom Komentar", value="Komentar")

    if st.button("🔄 Muat Ulang & Analisis Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.info("Data diperbarui secara otomatis setiap 5 menit. Klik tombol untuk pembaruan instan.")

st.title("📊 Dasbor Analisis Sentimen Komprehensif")
st.markdown("Menganalisis komentar dari Google Sheets menggunakan Model IndoBERT.")

# Main logic
if not NAMA_SPREADSHEET:
    st.info("Silakan masukkan Nama Google Spreadsheet di sidebar untuk memulai.")
else:
    try:
        gsheet_client = connect_to_gsheet()
        raw_df = fetch_data_from_gsheet(gsheet_client, NAMA_SPREADSHEET, NAMA_WORKSHEET, KOLOM_KOMENTAR)
        
        if raw_df.empty:
            st.warning("⚠️ Tidak ada data untuk dianalisis. Periksa konfigurasi sidebar atau isi spreadsheet Anda.")
        else:
            sentiment_model = load_sentiment_model()
            df = analyze_sentiment(raw_df.copy(), sentiment_model)

            tab1, tab2, tab3 = st.tabs(["📈 Ringkasan & Tren", "🔑 Analisis Kata Kunci", "📄 Jelajahi Data"])

            with tab1:
                st.header("Ringkasan Metrik Utama")
                col1, col2, col3, col4 = st.columns(4)
                sentiment_counts = df['sentimen'].value_counts()
                
                col1.metric("Total Komentar", len(df))
                col2.metric("👍 Komentar Positif", sentiment_counts.get('Positif', 0))
                col3.metric("👎 Komentar Negatif", sentiment_counts.get('Negatif', 0))
                col4.metric("😐 Komentar Netral", sentiment_counts.get('Netral', 0))

                st.header("Distribusi Sentimen")
                fig_pie = px.pie(df, names='sentimen', title='Persentase Sentimen', hole=0.4,
                                 color_discrete_map={'Positif':'green', 'Negatif':'red', 'Netral':'grey'},
                                 template='plotly_dark')
                st.plotly_chart(fig_pie, use_container_width=True)

                if 'tanggal' in df.columns:
                    st.header("Tren Sentimen Berdasarkan Tanggal")
                    df_tren = df.copy()
                    df_tren['tanggal_saja'] = df_tren['tanggal'].dt.date
                    sentimen_per_hari = df_tren.groupby(['tanggal_saja', 'sentimen']).size().unstack(fill_value=0)
                    
                    fig_tren = px.line(sentimen_per_hari, x=sentimen_per_hari.index, y=sentimen_per_hari.columns,
                                       title='Jumlah Komentar per Hari', markers=True,
                                       labels={'tanggal_saja': 'Tanggal', 'value': 'Jumlah Komentar', 'sentimen': 'Sentimen'},
                                       color_discrete_map={'Positif':'green', 'Negatif':'red', 'Netral':'grey'},
                                       template='plotly_dark')
                    st.plotly_chart(fig_tren, use_container_width=True)

            with tab2:
                st.header("Kata Kunci yang Paling Sering Muncul")
                df_positif = df[df['sentimen'] == 'Positif']['komentar']
                df_negatif = df[df['sentimen'] == 'Negatif']['komentar']
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Dari Komentar Positif")
                    create_wordcloud(df_positif, "")
                with col2:
                    st.subheader("Dari Komentar Negatif")
                    create_wordcloud(df_negatif, "")

            with tab3:
                st.header("Detail Data dan Hasil Analisis")
                st.dataframe(df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
        st.info("Pastikan konfigurasi di sidebar benar, file `secrets.toml` ada, dan spreadsheet sudah dibagikan dengan email service account.")
