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

# Download stopwords jika belum ada
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# --- KONFIGURASI HALAMAN DAN GAYA (CSS) ---
st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk menerapkan CSS kustom
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Anda bisa membuat file style.css atau langsung menaruh CSS di sini
# Untuk kemudahan, kita taruh langsung di sini
st.markdown("""
<style>
/* Latar Belakang Utama */
.stApp {
    background-color: #0d1117; /* Warna dasar gelap */
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
h1, h2, h3 {
    color: #c9d1d9;
}

/* Tombol */
.stButton>button {
    border: 2px solid #30363d;
    border-radius: 10px;
    color: #c9d1d9;
    background-color: #21262d;
}
.stButton>button:hover {
    border-color: #8b949e;
    color: #f0f6fc;
}
</style>
""", unsafe_allow_html=True)


# --- FUNGSI-FUNGSI UTAMA (KONEKSI, MODEL, ANALISIS) ---

@st.cache_resource
def connect_to_gsheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(creds)
    return client

@st.cache_data(ttl=300)
def fetch_data_from_gsheet(_gsheet_client, sheet_name, worksheet_name):
    st.toast("🔄 Mengambil data baru dari Google Sheets...")
    sheet = _gsheet_client.open(sheet_name).worksheet(worksheet_name)
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    # Pastikan kolom 'komentar' ada dan bertipe string
    if 'komentar' in df.columns:
        df = df[df['komentar'].astype(str).str.strip() != ''] # Hapus baris komentar kosong
    return df

@st.cache_resource
def load_sentiment_model():
    model = pipeline(
        "sentiment-analysis",
        model="indobenchmark/indobert-base-p2-sentiment-classifier"
    )
    return model

@st.cache_data
def analyze_sentiment(_df, _model):
    if 'komentar' not in _df.columns:
        st.error("Spreadsheet harus memiliki kolom bernama 'komentar'.")
        return pd.DataFrame() # Kembalikan DataFrame kosong jika tidak ada kolom 'komentar'

    texts = _df['komentar'].tolist()
    results = _model(texts)
    
    _df['sentimen'] = [res['label'] for res in results]
    _df['skor'] = [res['score'] for res in results]
    # Ubah label menjadi lebih mudah dibaca
    _df['sentimen'] = _df['sentimen'].replace({'positive': 'Positif', 'negative': 'Negatif', 'neutral': 'Netral'})
    return _df

# --- FUNGSI-FUNGSI VISUALISASI ---

def create_wordcloud(text_series, title):
    stopwords_indonesia = set(nltk.corpus.stopwords.words('indonesian'))
    text = ' '.join(text_series.astype(str).tolist())
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color=None, # Transparan
        colormap='viridis',
        stopwords=stopwords_indonesia,
        mode="RGBA"
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.title(title, color='white', fontsize=20)
    fig.patch.set_alpha(0) # Transparan
    st.pyplot(fig)

def create_ngram_barchart(text_series, n, title):
    stopwords_indonesia = set(nltk.corpus.stopwords.words('indonesian'))
    
    # Membersihkan teks
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    words = [word for text in text_series.apply(clean_text) for word in text.split() if word not in stopwords_indonesia]
    
    n_grams = ngrams(words, n)
    ngram_counts = Counter(n_grams)
    
    most_common_ngrams = ngram_counts.most_common(10)
    
    if not most_common_ngrams:
        st.write(f"Tidak cukup data untuk membuat grafik {title}.")
        return

    ngram_df = pd.DataFrame(most_common_ngrams, columns=['ngram', 'count'])
    ngram_df['ngram'] = ngram_df['ngram'].apply(lambda x: ' '.join(x))
    
    fig = px.bar(
        ngram_df, 
        x='count', 
        y='ngram', 
        orientation='h', 
        title=title,
        labels={'count': 'Jumlah', 'ngram': 'Frasa'},
        template='plotly_dark'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


# --- UI APLIKASI STREAMLIT ---

# Sidebar
with st.sidebar:
    st.title("⚙️ Kontrol Dasbor")
    st.markdown("Dasbor ini menganalisis sentimen dari komentar yang tersimpan di Google Sheets.")
    
    # GANTI DENGAN NAMA SPREADSHEET DAN WORKSHEET ANDA
    NAMA_SPREADSHEET = "Nama Spreadsheet Anda"  # <--- GANTI INI
    NAMA_WORKSHEET = "Sheet1"                # <--- GANTI INI (biasanya 'Sheet1')

    if st.button("🔄 Refresh Data & Analisis"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.info("Data diperbarui secara otomatis setiap 5 menit. Klik tombol untuk pembaruan instan.")


st.title("📊 Dasbor Analisis Sentimen Komprehensif")
st.markdown("Menggunakan Model IndoBERT untuk menganalisis komentar secara *real-time*.")

# Main logic
try:
    gsheet_client = connect_to_gsheet()
    raw_df = fetch_data_from_gsheet(gsheet_client, NAMA_SPREADSHEET, NAMA_WORKSHEET)
    
    if raw_df.empty or 'komentar' not in raw_df.columns:
        st.warning("⚠️ Spreadsheet kosong atau tidak memiliki kolom 'komentar'. Mohon isi data terlebih dahulu.")
    else:
        sentiment_model = load_sentiment_model()
        df = analyze_sentiment(raw_df.copy(), sentiment_model)

        # Membuat Tabs untuk layout yang rapi
        tab1, tab2, tab3 = st.tabs(["📈 Ringkasan Umum", "🔑 Analisis Kata Kunci", "📄 Jelajahi Data"])

        with tab1:
            st.header("Ringkasan Metrik Utama")
            col1, col2, col3, col4 = st.columns(4)
            sentiment_counts = df['sentimen'].value_counts()
            
            with col1:
                st.metric(label="Total Komentar Dianalisis", value=len(df))
            with col2:
                st.metric(label="👍 Komentar Positif", value=sentiment_counts.get('Positif', 0))
            with col3:
                st.metric(label="👎 Komentar Negatif", value=sentiment_counts.get('Negatif', 0))
            with col4:
                st.metric(label="😐 Komentar Netral", value=sentiment_counts.get('Netral', 0))

            st.header("Distribusi Sentimen")
            fig_pie = px.pie(
                df, 
                names='sentimen', 
                title='Persentase Sentimen Komentar',
                hole=0.4,
                color_discrete_map={'Positif':'green', 'Negatif':'red', 'Netral':'grey'}
            )
            fig_pie.update_layout(template='plotly_dark')
            st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.header("Kata Kunci yang Paling Sering Muncul")
            
            df_positif = df[df['sentimen'] == 'Positif']['komentar']
            df_negatif = df[df['sentimen'] == 'Negatif']['komentar']

            col1, col2 = st.columns(2)
            with col1:
                if not df_positif.empty:
                    create_wordcloud(df_positif, "Word Cloud Sentimen Positif")
                else:
                    st.write("Tidak ada data sentimen positif untuk Word Cloud.")
            
            with col2:
                if not df_negatif.empty:
                    create_wordcloud(df_negatif, "Word Cloud Sentimen Negatif")
                else:
                    st.write("Tidak ada data sentimen negatif untuk Word Cloud.")
            
            st.header("Frasa Umum (Bigram)")
            col3, col4 = st.columns(2)
            with col3:
                if not df_positif.empty:
                    create_ngram_barchart(df_positif, 2, "Frasa Positif Paling Umum")
                else:
                    st.write("Tidak ada data sentimen positif untuk Analisis Frasa.")
            with col4:
                if not df_negatif.empty:
                    create_ngram_barchart(df_negatif, 2, "Frasa Negatif Paling Umum")
                else:
                    st.write("Tidak ada data sentimen negatif untuk Analisis Frasa.")

        with tab3:
            st.header("Detail Data dan Hasil Analisis")
            st.markdown("Anda dapat mencari dan mengurutkan data di bawah ini.")
            st.dataframe(df, use_container_width=True)


except Exception as e:
    st.error(f"❌ Terjadi kesalahan: {e}")
    st.info("Pastikan konfigurasi `secrets.toml` benar dan spreadsheet sudah dibagikan dengan email service account.")
