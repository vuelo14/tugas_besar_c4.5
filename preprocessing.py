import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# ==============================================================================
# MODUL P5.3: PREPROCESSING DATA TEKS
# Penanggung Jawab: Wawan Siswanto & Ahmadi
# Tujuan: Membersihkan data mentah agar siap dilatih oleh AI (C4.5)
# ==============================================================================

# 1. SETUP LIBRARY NLTK (Download kamus kata hubung jika belum ada)
print("⏳ Sedang menyiapkan library NLTK...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Definisi Stopwords Bahasa Indonesia (dan, yang, di, ke, dari, dll)
stop_words = set(stopwords.words('indonesian'))

# Tambahkan kata-kata sampah (slang) yang sering muncul di sosmed tapi tidak penting
custom_stopwords = {
    'yg', 'dg', 'rt', 'dgn', 'ny', 'kalo', 'klo', 'tuh', 'sih', 'nya', 
    'gak', 'ga', 'aja', 'sdh', 'udah', 'dah', 'sy', 'gw', 'loe'
}
stop_words.update(custom_stopwords)

# ==============================================================================
# 2. FUNGSI CLEANING
# ==============================================================================
def clean_text(text):
    """
    Fungsi untuk membersihkan satu kalimat/paragraf teks.
    """
    # Pastikan input berupa string (cegah error jika ada data angka/kosong)
    text = str(text)
    
    # a. Case Folding (Ubah ke huruf kecil semua)
    text = text.lower()
    
    # b. Hapus URL/Link (http://... atau www...)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # c. Hapus Mention (@username) dan Hashtag (#)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # d. Hapus Angka dan Tanda Baca (Hanya simpan huruf a-z)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # e. Hapus Spasi Berlebih (Double space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # f. Stopword Removal (Hapus kata hubung)
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    
    # Gabungkan kembali menjadi kalimat
    return " ".join(filtered_words)

# ==============================================================================
# 3. PROSES UTAMA
# ==============================================================================
def main():
    nama_file_input = 'dataset_final_gabungan.csv'
    nama_file_output = 'dataset_bersih_siap_training.csv'
    
    print(f"\n📂 Membaca file: {nama_file_input} ...")
    
    try:
        df = pd.read_csv(nama_file_input)
    except FileNotFoundError:
        print("❌ ERROR: File tidak ditemukan!")
        print(f"   Pastikan file '{nama_file_input}' ada di folder yang sama.")
        return

    # LOGIKA DETEKSI KOLOM TEKS
    # Kita cari kolom yang namanya mirip 'content', 'narasi', atau 'isi'
    target_col = None
    calon_kolom = ['content', 'narasi', 'isi', 'text', 'judul', 'title']
    
    for col in df.columns:
        if col.lower() in calon_kolom:
            target_col = col
            break
    
    # Jika tidak ketemu nama yang pas, pakai kolom index ke-1 (biasanya setelah nomor/judul)
    if target_col is None:
        target_col = df.columns[1] 
    
    print(f"✅ Kolom teks terdeteksi: '{target_col}'")
    print(f"📊 Jumlah data awal: {len(df)} baris")
    print("⏳ Sedang melakukan cleaning (ini memakan waktu 1-3 menit)...")

    # TERAPKAN CLEANING KE SELURUH DATA
    # Hasilnya disimpan di kolom baru bernama 'text_clean'
    df['text_clean'] = df[target_col].apply(clean_text)
    
    # Hapus data yang hasil cleaningnya kosong (misal isinya cuma link doang)
    df = df[df['text_clean'] != '']
    
    # Simpan File Baru
    print(f"💾 Menyimpan hasil ke: {nama_file_output} ...")
    df.to_csv(nama_file_output, index=False)
    
    print("\n" + "="*50)
    print("🎉 PREPROCESSING SELESAI!")
    print("="*50)
    print("Sekarang kolom 'text_clean' SUDAH ADA.")
    print("Silakan jalankan kembali script 'training_c45.py'.")

if __name__ == "__main__":
    main()