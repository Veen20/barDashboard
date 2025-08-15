import streamlit as st
import pandas as pd
import plotly.express as px

# =====================
# Konfigurasi Halaman
# =====================
st.set_page_config(
    page_title="Samsat Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# CSS untuk Tema Modern
# =====================
st.markdown("""
    <style>
        /* Warna Background dan Font */
        .main {
            background-color: #0e1117;
            color: white;
            font-family: 'Inter', sans-serif;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #111827;
        }
        /* Judul Metric */
        .metric-title {
            font-size: 14px;
            color: rgba(255,255,255,0.6);
            text-transform: uppercase;
            letter-spacing: .06em;
        }
        /* Card Style */
        .stMetric {
            background-color: #1f2937;
            padding: 15px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# =====================
# Data Dummy untuk Contoh
# =====================
data = {
    "Tanggal": pd.date_range(start="2025-07-20", periods=10, freq="D"),
    "Sentimen": ["Positif", "Negatif", "Netral", "Positif", "Netral", "Negatif", "Positif", "Netral", "Positif", "Negatif"],
    "Skor": [0.9, 0.3, 0.5, 0.8, 0.55, 0.2, 0.87, 0.6, 0.92, 0.4]
}
df = pd.DataFrame(data)

# =====================
# Sidebar Navigation
# =====================
menu = st.sidebar.radio("Navigasi", ["📊 Overview", "📈 Analisis", "🌍 Visualisasi"])

# =====================
# Overview
# =====================
if menu == "📊 Overview":
    st.title("📊 Samsat Sentiment - Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Komentar Masuk", len(df), "+5 hari ini")
    with col2:
        st.metric("Sentimen Positif", f"{(df['Sentimen']=='Positif').mean()*100:.1f}%", "↑")
    with col3:
        st.metric("Rata-rata Skor", f"{df['Skor'].mean():.2f}", "")

    # Grafik Garis Skor
    fig = px.line(df, x="Tanggal", y="Skor", markers=True, title="Perubahan Skor Sentimen",
                  template="plotly_dark", color_discrete_sequence=["#00f5d4"])
    st.plotly_chart(fig, use_container_width=True)

# =====================
# Analisis
# =====================
elif menu == "📈 Analisis":
    st.title("📈 Analisis Sentimen")
    fig = px.histogram(df, x="Sentimen", color="Sentimen",
                       title="Distribusi Sentimen",
                       template="plotly_dark",
                       color_discrete_sequence=["#00f5d4", "#ff595e", "#ffca3a"])
    st.plotly_chart(fig, use_container_width=True)

# =====================
# Visualisasi
# =====================
elif menu == "🌍 Visualisasi":
    st.title("🌍 Visualisasi Lokasi atau Kategori")
    st.info("Di sini kamu bisa menambahkan peta atau grafik kategori terkait Samsat.")

