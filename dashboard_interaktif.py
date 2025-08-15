import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(page_title="Dashboard Analisis Sentimen", layout="wide")

# =========================
# CSS Custom
# =========================
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1, h2, h3 {
        color: #2E86C1;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Sidebar Navigasi
# =========================
with st.sidebar:
    selected = option_menu(
        "Navigasi",
        ["📊 Overview", "📈 Analisis Sentimen", "📉 Visualisasi"],
        icons=["house", "chat-left-text", "bar-chart"],
        menu_icon="cast",
        default_index=0,
    )

# =========================
# Dataset Contoh
# =========================
@st.cache_data
def load_sample_data():
    data = {
        "tanggal": pd.date_range(start="2025-01-01", periods=12, freq="D"),
        "komentar": [
            "Pelayanan sangat cepat", "Website sering error", "Cukup memuaskan",
            "Lambat dalam memproses", "Sangat membantu", "Biasa saja",
            "Petugas ramah", "Sulit diakses", "Pengalaman menyenangkan",
            "Aplikasi sering crash", "Cepat dan praktis", "Kurang memuaskan"
        ],
        "sentimen": [
            "positif", "negatif", "netral", "negatif", "positif", "netral",
            "positif", "negatif", "positif", "negatif", "positif", "negatif"
        ]
    }
    return pd.DataFrame(data)

# =========================
# Upload atau Load Data
# =========================
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Menggunakan dataset contoh.")
    df = load_sample_data()

# Pastikan kolom sesuai
if not all(col in df.columns for col in ["tanggal", "komentar", "sentimen"]):
    st.error("Dataset harus memiliki kolom: tanggal, komentar, sentimen")
    st.stop()

df["tanggal"] = pd.to_datetime(df["tanggal"])

# =========================
# Halaman Overview
# =========================
if selected == "📊 Overview":
    st.markdown("<h1>📊 Dashboard Analisis Sentimen</h1>", unsafe_allow_html=True)
    st.write("Dashboard ini menampilkan ringkasan sentimen dari komentar masyarakat.")

    total = len(df)
    positif = (df["sentimen"] == "positif").sum()
    netral = (df["sentimen"] == "netral").sum()
    negatif = (df["sentimen"] == "negatif").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Komentar", total)
    col2.metric("Positif", positif)
    col3.metric("Netral", netral)
    col4.metric("Negatif", negatif)

    with st.expander("📄 Lihat Data Awal"):
        st.dataframe(df.head())

# =========================
# Halaman Analisis Sentimen
# =========================
elif selected == "📈 Analisis Sentimen":
    st.header("📈 Analisis Sentimen")

    # Filter tanggal
    date_min = df["tanggal"].min()
    date_max = df["tanggal"].max()
    start_date, end_date = st.date_input("Pilih Rentang Tanggal", [date_min, date_max])
    df_filtered = df[(df["tanggal"] >= pd.to_datetime(start_date)) & (df["tanggal"] <= pd.to_datetime(end_date))]

    # Distribusi sentimen
    sent_count = df_filtered["sentimen"].value_counts().reset_index()
    sent_count.columns = ["sentimen", "jumlah"]

    fig_pie = px.pie(sent_count, names="sentimen", values="jumlah", title="Distribusi Sentimen", color="sentimen",
                     color_discrete_map={"positif": "#2ECC71", "netral": "#F4D03F", "negatif": "#E74C3C"})
    st.plotly_chart(fig_pie, use_container_width=True)

    # Wordcloud
    st.subheader("Word Cloud per Sentimen")
    col1, col2, col3 = st.columns(3)
    for senti, col in zip(["positif", "netral", "negatif"], [col1, col2, col3]):
        text = " ".join(df_filtered[df_filtered["sentimen"] == senti]["komentar"])
        if text:
            wc = WordCloud(width=300, height=300, background_color="white", colormap="viridis").generate(text)
            col.image(wc.to_array(), caption=f"Word Cloud {senti}")
        else:
            col.write(f"Tidak ada komentar {senti}.")

# =========================
# Halaman Visualisasi
# =========================
elif selected == "📉 Visualisasi":
    st.header("📉 Visualisasi Trend Sentimen")

    # Trend sentimen per tanggal
    trend = df.groupby(["tanggal", "sentimen"]).size().reset_index(name="jumlah")
    fig_line = px.line(trend, x="tanggal", y="jumlah", color="sentimen", title="Trend Sentimen Harian",
                       color_discrete_map={"positif": "#2ECC71", "netral": "#F4D03F", "negatif": "#E74C3C"})
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("📄 Tabel Komentar")
    st.dataframe(df)

    # Download tombol
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    st.download_button(
        label="💾 Download CSV",
        data=buffer.getvalue(),
        file_name="sentimen_filtered.csv",
        mime="text/csv"
    )

