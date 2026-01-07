import joblib
import re
import os
import nltk
from nltk.corpus import stopwords

# ==============================================================================
# APLIKASI UTAMA DETEKSI HOAX (VERSION 3.0 - FINAL)
# Penanggung Jawab: Wawan Siswanto
# ==============================================================================

# 1. FUNGSI CLEANING (Harus sama persis dengan saat training)
# ------------------------------------------------------------------------------
def clean_text_input(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Stopwords
    try:
        stop_words = set(stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('indonesian'))
        
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

# 2. LOAD MODEL & VECTORIZER
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("🤖 MEMUAT SISTEM KECERDASAN BUATAN...")
print("="*50)

file_model = 'model_c45_hoax.pkl'
file_vocab = 'vectorizer_tfidf.pkl'

if not os.path.exists(file_model) or not os.path.exists(file_vocab):
    print("❌ ERROR CRITICAL: File model tidak ditemukan!")
    print("   Pastikan 'model_c45_hoax.pkl' dan 'vectorizer_tfidf.pkl' ada.")
    print("   Jalankan training_c45.py terlebih dahulu.")
    exit()

try:
    clf = joblib.load(file_model)
    tfidf = joblib.load(file_vocab)
    print("✅ Model C4.5 & TF-IDF Vectorizer berhasil dimuat.")
except Exception as e:
    print(f"❌ Terjadi kesalahan saat memuat model: {e}")
    exit()

# 3. INTERFACE PENGGUNA (LOOP)
# ------------------------------------------------------------------------------
print("\nSistem Siap! Ketik 'exit' atau 'keluar' untuk berhenti.")

while True:
    print("\n" + "-"*30)
    input_user = input("📝 Masukkan Judul/Isi Berita: ")
    
    if input_user.lower() in ['exit', 'keluar', 'quit']:
        print("👋 Terima kasih telah menggunakan sistem ini.")
        break
    
    if len(input_user) < 5:
        print("⚠️ Teks terlalu pendek. Masukkan berita yang lebih lengkap.")
        continue

    # A. Preprocessing Input
    text_clean = clean_text_input(input_user)
    
    # B. Transformasi ke Angka (Vectorization)
    text_vector = tfidf.transform([text_clean]).toarray()
    
    # C. Prediksi
    # Hasil prediksi: 0 = Fakta, 1 = Hoax
    prediksi_kelas = clf.predict(text_vector)[0]
    # Probabilitas: [kemungkinan_fakta, kemungkinan_hoax]
    probabilitas = clf.predict_proba(text_vector)[0] 
    
    # D. Tampilkan Hasil
    label_hasil = "HOAX 🚨" if prediksi_kelas == 1 else "FAKTA ✅"
    confidence = probabilitas[prediksi_kelas] * 100 # Ambil persentase keyakinan
    
    print(f"\n📢 HASIL ANALISIS AI:")
    print(f"   Status     : {label_hasil}")
    print(f"   Tingkat Keyakinan : {confidence:.2f}%")
    
    # Analisis Tambahan (Kenapa Hoax?)
    print(f"   Kata Kunci Terdeteksi (Clean): '{text_clean}'")