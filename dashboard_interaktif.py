import streamlit as st
import os

# -----------------------------
# Load CSS modern
# -----------------------------
def load_css(file_path="style.css"):
    """Load CSS file lokal"""
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"File CSS tidak ditemukan: {file_path}")

load_css("style.css")  # otomatis load modern style + animasi

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Dashboard Interaktif",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar modern
# -----------------------------
with st.sidebar:
    st.title("Menu Sidebar")
    pilihan = st.radio("Pilih Visualisasi", ["Home", "Data", "Analisis"])
    st.button("Refresh Data")

# -----------------------------
# Dark mode toggle
# -----------------------------
dark_mode = st.checkbox("Dark mode", value=False)
if dark_mode:
    st.markdown("<script>document.querySelector('body').classList.add('dark-mode')</script>", unsafe_allow_html=True)
else:
    st.markdown("<script>document.querySelector('body').classList.remove('dark-mode')</script>", unsafe_allow_html=True)

# -----------------------------
# Konten utama dengan animasi
# -----------------------------
st.markdown(
    f"""
    <div class="hero">
      <h1>🧠 Dashboard Sentimen — Real-time (Google Sheets)</h1>
      <p>Preprocessing otomatis • IndoBERT sentiment • Visual & insight • Download hasil.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# contoh metric cards
cols = st.columns(3)
for col in cols:
    col.markdown(
        f'<div class="metric-card"><div class="metric-title">Contoh</div><div class="metric-value">123</div></div>',
        unsafe_allow_html=True
    )

# -----------------------------
# Footer modern
# -----------------------------
st.markdown(
    "<footer>&copy; 2025 Dashboard Interaktif e-SIGNAL / SAMSAT • Created with ❤️</footer>",
    unsafe_allow_html=True
)
