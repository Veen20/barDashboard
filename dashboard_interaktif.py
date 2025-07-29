import streamlit as st
import pandas as pd
import gspread
import json
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import torch
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------
# KONFIGURASI DASHBOARD
# ----------------------------
st.set_page_config(page_title="📊 Dashboard Analisis Transaksi & Sentimen Masyarakat Terhadap UPTB Samsat Palembang 1", layout="wide")
st.title("📊  Dashboard Analisis Transaksi & Sentimen Masyarakat Terhadap UPTB Samsat Palembang 1")

# ----------------------------
# AUTENTIKASI GSPREAD
# ----------------------------
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(st.secrets["GOOGLE_SHEET_CREDENTIALS"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

# ----------------------------
# AMBIL DATA DARI SPREADSHEET
# ----------------------------
spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1NaV3vKTTRohc5DMdxM807S2sUkrNGobD0Tt0Bu9Uqx0/edit?usp=sharing")
sheet_transaksi = spreadsheet.worksheet("transaksi")
sheet_komentar = spreadsheet.worksheet("komentar")

df_trans = pd.DataFrame(sheet_transaksi.get_all_records())
df_komentar = pd.DataFrame(sheet_komentar.get_all_records())

# ----------------------------
# TAB DASHBOARD
# ----------------------------
tab1, tab2, tab3 = st.tabs(["💳 Data Transaksi", "💬 Data Komentar", "📈 Visualisasi Gabungan"])

with tab1:
    st.subheader("📌 Tabel Transaksi")
    st.dataframe(df_trans)

    # Preprocessing Tanggal & Jam
    df_trans['TANGGAL'] = pd.to_datetime(df_trans['TANGGAL'])
    df_trans['datetime'] = pd.to_datetime(df_trans['TANGGAL'].dt.date.astype(str) + ' ' + df_trans['JAM'].astype(str), errors='coerce')
    df_trans.dropna(subset=['datetime'], inplace=True)
    df_trans['jam_only'] = df_trans['datetime'].dt.hour
    hari_mapping = {
        'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
        'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
    }
    df_trans['hari'] = df_trans['datetime'].dt.day_name().map(hari_mapping)

    # Visualisasi
    st.markdown("### 📊 Visualisasi")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Total Transaksi per Hari**")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='hari', data=df_trans, order=['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'], ax=ax1)
        ax1.set_ylabel("Jumlah Transaksi")
        ax1.set_xlabel("Hari")
        st.pyplot(fig1)

    with col2:
        st.markdown("**Distribusi Jam Transaksi**")
        # Pastikan kolom 'jam_only' bertipe integer
        df_trans['jam_only'] = df_trans['jam_only'].astype(int)
    
        # Buat plot
        fig2, ax2 = plt.subplots(figsize=(14, 11))
        sns.histplot(
            df_trans['jam_only'],
            bins=24,
            binrange=(0, 24),
            kde=True,
            color='dodgerblue',
            ax=ax2)
    
        # Atur label sumbu X dari 00 sampai 23
        ax2.set_xticks(np.arange(0, 24, 1))
        ax2.set_xticklabels([f"{i:02d}" for i in range(24)])
        ax2.set_xlabel("Jam Transaksi (00-23)")
    
        # Tampilkan plot di Streamlit
        st.pyplot(fig2)
      
        
    # Clustering Waktu
    bins = [-1, 5, 10, 15, 23]
    labels = ['Dini Hari', 'Pagi', 'Siang', 'Sore-Malam']
    df_trans['kategori_waktu'] = pd.cut(df_trans['jam_only'], bins=bins, labels=labels)

    df_hari = pd.crosstab(df_trans['datetime'].dt.date, df_trans['kategori_waktu'])
    df_hari['Total Harian'] = df_hari.sum(axis=1)
    scaler = StandardScaler()
    X_hari = scaler.fit_transform(df_hari)
    kmeans_hari = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_hari['Profil Hari'] = kmeans_hari.fit_predict(X_hari)
    df_hari['Nama Profil'] = df_hari['Profil Hari'].map({
        0: "Hari Sangat Tenang", 1: "Hari Sangat Sibuk", 2: "Hari Normal"
    })

   # Ambil nama hari dari index tanggal
    df_hari['Hari'] = pd.to_datetime(df_hari.index).day_name()
    
    # Mapping ke Bahasa Indonesia
    hari_mapping = {
        'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
        'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
    }
    df_hari['Hari'] = df_hari['Hari'].map(hari_mapping)
    
    # Urutkan nama hari
    urutan_hari = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    df_hari['Hari'] = pd.Categorical(df_hari['Hari'], categories=urutan_hari, ordered=True)
    
    # Crosstab distribusi jumlah profil per nama hari
    df_profil_hari = pd.crosstab(df_hari['Hari'], df_hari['Nama Profil'])

       
    st.markdown("**Rata-Rata Jumlah Transaksi per Kategori Waktu untuk Tiap Profil Hari**")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    df_profil_hari.plot(kind='bar', stacked=False, ax=ax5, colormap='Set2')

    ax5.set_title("Distribusi Profil Hari")
    ax5.set_xlabel("Hari")
    ax5.set_ylabel("Frekuensi")
    ax5.legend(title="Nama Profil")
    ax5.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig5)
    


with tab2:
    st.subheader("📝 Komentar Pengguna")
    st.dataframe(df_komentar)

    # Preprocessing + Sentiment Analysis
    df_komentar['komentar_bersih'] = df_komentar['Komentar'].astype(str).apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))

    def sentimen_manual(text):
        positif = ['mantap', 'oke', 'bagus', 'cepat', 'praktis', 'baik', 'mantabb', 'terima kasih', 'top']
        negatif = ['gagal', 'error', 'tidak bisa', 'jelek', 'lama', 'rusak', 'tidak dapat', 'buruk', 'tidak muncul']
        if any(p in text for p in positif): return 2
        elif any(n in text for n in negatif): return 0
        return None

    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = AutoModelForSequenceClassification.from_pretrained("mdhugol/indonesia-bert-sentiment-classification")

    def predict_sentiment(text):
        rule = sentimen_manual(text)
        if rule is not None:
            return rule
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        return torch.argmax(probs).item()

    df_komentar['label_sentimen'] = df_komentar['komentar_bersih'].apply(predict_sentiment)
    df_komentar['kategori_sentimen'] = df_komentar['label_sentimen'].map({0: 'Negatif', 1: 'Netral', 2: 'Positif'})
    df_komentar['tanggal'] = pd.to_datetime(df_komentar['Tanggal'], errors='coerce')
    df_komentar['hari'] = df_komentar['tanggal'].dt.day_name()


    # Tampilkan beberapa komentar terbaru
    st.markdown("### 🔍 Komentar Terbaru")
    for komentar in df_komentar['Komentar'].head(5):
        st.write(f"🗨️ {komentar}")

    # Visualisasi Sentimen
    st.markdown("### 📊 Visualisasi Sentimen")
    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots()
        df_komentar['kategori_sentimen'].value_counts().plot(kind='bar', color=['red', 'gray', 'green'], ax=ax3)
        ax3.set_ylabel("Jumlah Komentar")
        ax3.set_xlabel("Kategori Sentimen")
        ax3.set_title("Distribusi Sentimen")
        st.pyplot(fig3)

    with col4:   
        st.markdown("**Distribusi Sentimen per Hari**")
    
        # Mapping hari
        hari_mapping = {
            'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
            'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
        }
    
        df_komentar['hari'] = df_komentar['tanggal'].dt.day_name().map(hari_mapping)
    
        # Definisikan urutan
        semua_hari = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
        semua_sentimen = ['Negatif', 'Netral', 'Positif']
    
        # Ubah jadi kategori supaya urut
        df_komentar['hari'] = pd.Categorical(df_komentar['hari'], categories=semua_hari, ordered=True)
        df_komentar['kategori_sentimen'] = pd.Categorical(df_komentar['kategori_sentimen'], categories=semua_sentimen, ordered=True)
    
        # Buat kombinasi semua hari x sentimen
        index_kombinasi = pd.MultiIndex.from_product([semua_hari, semua_sentimen], names=['hari', 'kategori_sentimen'])
    
        # Hitung dan isi 0 kalau kosong
        distribusi_sentimen = (
            df_komentar.groupby(['hari', 'kategori_sentimen'])
            .size()
            .reindex(index_kombinasi, fill_value=0)
            .unstack()
        )
    
        # Plot
        plt.figure(figsize=(10,6))
        sentimen_hari.plot(kind='bar', stacked=True, colormap='Set2')
        plt.title("Distribusi Sentimen per Hari", fontsize=14, weight='bold')
        plt.xlabel("Hari")
        plt.ylabel("Jumlah Komentar")
        plt.xticks(rotation=45)
        plt.legend(title="Kategori Sentimen")
        plt.tight_layout()
        plt.grid(axis='y')
        plt.show()



with tab3:
    st.subheader("📈 Ringkasan Gabungan")
    st.write("Jumlah total transaksi:", len(df_trans))
    st.write("Jumlah komentar:", len(df_komentar))
    st.write("Distribusi kategori waktu transaksi:")
    st.dataframe(df_trans['kategori_waktu'].value_counts())
    st.write("Distribusi sentimen:")
    st.dataframe(df_komentar['kategori_sentimen'].value_counts())
