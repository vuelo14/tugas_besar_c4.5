import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# ==============================================================================
# MODUL VISUALISASI POHON KEPUTUSAN
# Output: Gambar 'struktur_pohon_c45.png' untuk Laporan Bab 5
# ==============================================================================

print("⏳ Sedang memuat model...")
try:
    clf = joblib.load('model_c45_hoax.pkl')
    tfidf = joblib.load('vectorizer_tfidf.pkl')
except:
    print("❌ Model tidak ditemukan. Jalankan training dulu.")
    exit()

# Mengambil nama fitur (kata-kata) dari TF-IDF
# Agar pohonnya menampilkan kata (misal: "jokowi <= 0.5") bukan angka (misal: "X[20] <= 0.5")
feature_names = tfidf.get_feature_names_out()

print("🌳 Sedang menggambar pohon keputusan (Mungkin butuh 1 menit)...")

# Kita batasi kedalaman (max_depth) saat visualisasi agar gambar terbaca
# Kalau ditampilkan semua (depth=20) gambarnya akan benang kusut
plt.figure(figsize=(20, 10), dpi=300)

plot_tree(clf, 
          max_depth=3,              # Hanya tampilkan 3 level teratas (Root & Cabang utama)
          feature_names=feature_names, 
          class_names=['Fakta', 'Hoax'],
          filled=True,              # Beri warna (Biru=Fakta, Oranye=Hoax)
          rounded=True, 
          fontsize=10)

plt.title("Visualisasi 3 Level Teratas Decision Tree C4.5 Deteksi Hoax")
plt.savefig('struktur_pohon_c45.png') # Simpan gambar
plt.show()

print("✅ Gambar tersimpan: 'struktur_pohon_c45.png'")
print("💡 Silakan masukkan gambar ini ke Laporan Bab 5 (Analisis Hasil).")