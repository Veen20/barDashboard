import streamlit as st
import base64

def load_css_with_bg(css_file, image_file):
    # Baca gambar → base64
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    image_url = f"data:image/png;base64,{encoded}"

    # Baca file CSS dan ganti placeholder BACKGROUND_IMAGE
    with open(css_file) as f:
        css = f.read().replace("BACKGROUND_IMAGE", image_url)

    # Inject CSS ke Streamlit
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ==== Load CSS + Background ====
load_css_with_bg("style.css", "samsat.jpg")

# ==== Contoh konten dashboard ====
st.title("📊 Dashboard Analisis Samsat")
st.write("Sekarang pakai `style.css` eksternal, background tetap muncul karena gambar disisipkan otomatis.")
st.button("Klik Saya")
