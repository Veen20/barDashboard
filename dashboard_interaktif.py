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
st.set_page_config(page_title="📊 Dashboard Transaksi & Sentimen eSIGNAL", layout="wide")
st.title("📊 Dashboard Transaksi & Sentimen eSIGNAL")

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
        plt.figure(figsize=(12, 6))
        sns.histplot(df['jam_only'], bins=24, kde=True, color='dodgerblue')
        plt.title('Distribusi Jam Transaksi')
        plt.xlabel('Jam (0-23)')
        plt.ylabel('Frekuensi')
        plt.grid(axis='y', linestyle='--')
        plt.xticks(range(24))
        plt.show()fig2, ax2 = plt.subplots()
    

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
        sentimen_hari = df_komentar.groupby(['hari', 'kategori_sentimen']).size().unstack().fillna(0)
        fig4, ax4 = plt.subplots()
        sentimen_hari.plot(kind='bar', stacked=True, colormap='Set2', ax=ax4)
        ax4.set_ylabel("Jumlah Komentar")
        ax4.set_xlabel("Hari")
        ax4.set_title("Distribusi Sentimen per Hari")
        st.pyplot(fig4)

with tab3:
    st.subheader("📈 Ringkasan Gabungan")
    st.write("Jumlah total transaksi:", len(df_trans))
    st.write("Jumlah komentar:", len(df_komentar))
    st.write("Distribusi kategori waktu transaksi:")
    st.dataframe(df_trans['kategori_waktu'].value_counts())
    st.write("Distribusi sentimen:")
    st.dataframe(df_komentar['kategori_sentimen'].value_counts())
