import streamlit as st
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords

# ==============================================================================
# CONFIG & SETUP
# ==============================================================================
st.set_page_config(
    page_title="Anti-Hoax Detector",
    page_icon="🛡️",
    layout="centered"
)

# Load NLTK resources
nltk.download('stopwords', quiet=True)

# Fungsi Cleaning (Sama seperti sebelumnya)
def clean_text_input(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        stop_words = set(stopwords.words('indonesian'))
    except:
        stop_words = set() # Fallback jika gagal load
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

# ==============================================================================
# LOAD MODEL (Cached agar cepat)
# ==============================================================================
@st.cache_resource
def load_model():
    try:
        clf = joblib.load('model_c45_hoax.pkl')
        tfidf = joblib.load('vectorizer_tfidf.pkl')
        return clf, tfidf
    except:
        return None, None

clf, tfidf = load_model()

# ==============================================================================
# TAMPILAN WEBSITE (UI)
# ==============================================================================
st.title("🛡️ Sistem Deteksi Berita Hoax")
st.markdown("""
Aplikasi ini menggunakan **Artificial Intelligence (Decision Tree C4.5)** untuk menganalisis apakah sebuah berita terindikasi **HOAX** atau **FAKTA**.
""")

st.write("---")

if clf is None:
    st.error("❌ Model tidak ditemukan! Harap jalankan `training_c45.py` terlebih dahulu.")
else:
    # Input User
    news_text = st.text_area("Masukkan Judul atau Isi Berita di sini:", height=150, placeholder="Contoh: Jokowi membagikan uang 500 triliun viralkan...")

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analisis Berita", type="primary")

    # Logika Prediksi
    if analyze_btn:
        if len(news_text) < 5:
            st.warning("⚠️ Teks terlalu pendek. Mohon masukkan berita yang lebih lengkap.")
        else:
            with st.spinner('Sedang menganalisis pola kata...'):
                # 1. Preprocessing
                clean_txt = clean_text_input(news_text)
                
                # 2. Vectorization
                text_vector = tfidf.transform([clean_txt]).toarray()
                
                # 3. Prediksi
                prediksi = clf.predict(text_vector)[0]
                probs = clf.predict_proba(text_vector)[0]
                confidence = probs[prediksi] * 100

                # 4. Tampilkan Hasil
                st.write("---")
                st.subheader("Hasil Analisis:")

                if prediksi == 1:
                    st.error(f"🚨 **TERINDIKASI HOAX**")
                    st.progress(confidence / 100, text=f"Tingkat Keyakinan AI: {confidence:.2f}%")
                    st.info("💡 **Saran:** Jangan sebarkan berita ini. Cek kembali di situs berita resmi.")
                else:
                    st.success(f"✅ **TERINDIKASI FAKTA**")
                    st.progress(confidence / 100, text=f"Tingkat Keyakinan AI: {confidence:.2f}%")
                    st.info("💡 **Info:** Pola bahasa berita ini mirip dengan berita valid pada umumnya.")

                # Debugging (Opsional - Bisa disembunyikan)
                with st.expander("Lihat Detail Teknis (Opsional)"):
                    st.text(f"Raw Input: {news_text}")
                    st.text(f"Clean Input: {clean_txt}")
                    st.write("Decision Tree Probability:", probs)