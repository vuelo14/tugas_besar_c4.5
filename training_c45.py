import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================================================================
# MODUL P6.1: IMPLEMENTASI CORE ALGORITHM C4.5 (VERSION 2.0)
# Penanggung Jawab: Wawan Siswanto
# ==============================================================================

print("🚀 MEMULAI PROSES TRAINING MODEL C4.5...")

# 1. LOAD DATASET BERSIH
# ------------------------------------------------------------------
# Kita pakai dataset hasil preprocessing di tahap sebelumnya
try:
    df = pd.read_csv('dataset_bersih_siap_training.csv')
    print(f"✅ Dataset dimuat: {len(df)} baris data.")
except FileNotFoundError:
    print("❌ Error: File 'dataset_bersih_siap_training.csv' tidak ditemukan.")
    exit()

# Pastikan tidak ada nilai null di kolom text_clean
df['text_clean'] = df['text_clean'].fillna('')

# 2. FEATURE EXTRACTION (TF-IDF)
# ------------------------------------------------------------------
# Mengubah teks menjadi vektor angka agar bisa dihitung matematikanya.
# max_features=5000 artinya kita hanya mengambil 5000 kata terpenting (agar tidak lemot)
print("⏳ Sedang melakukan pembobotan kata (TF-IDF)...")
tfidf = TfidfVectorizer(max_features=5000)

# X adalah Fitur (Data Teks yang sudah jadi angka)
X = tfidf.fit_transform(df['text_clean']).toarray()
# y adalah Target (Label: 0 atau 1)
y = df['label']

print(f"✅ Transformasi selesai. Ukuran Matrix: {X.shape}")

# 3. SPLIT DATA (80% Training, 20% Testing)
# ------------------------------------------------------------------
print("⏳ Membagi data Training (80%) dan Testing (20%)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. IMPLEMENTASI ALGORITMA DECISION TREE (C4.5)
# ------------------------------------------------------------------
# criterion='entropy' -> Menandakan kita menggunakan logika C4.5 (Information Gain)
# max_depth=20 -> Membatasi kedalaman pohon agar tidak overfitting (terlalu menghapal)
print("⏳ Sedang melatih model Decision Tree (ini mungkin butuh waktu)...")

clf = DecisionTreeClassifier(
    criterion='entropy',  # Kunci algoritma C4.5
    max_depth=20,         # Pembatasan kedalaman pohon
    random_state=42
)

clf.fit(X_train, y_train)

# 5. EVALUASI AWAL (UNIT TESTING SKALA BESAR)
# ------------------------------------------------------------------
print("\n--- HASIL TRAINING ---")
y_pred = clf.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

print(f"🎯 Akurasi Model: {akurasi * 100:.2f}%")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=['Fakta', 'Hoax']))

# 6. SIMPAN MODEL (SERIALIZATION)
# ------------------------------------------------------------------
# Menyimpan otak buatan (model) dan kamus kata (tfidf) agar bisa dipakai nanti
joblib.dump(clf, 'model_c45_hoax.pkl')
joblib.dump(tfidf, 'vectorizer_tfidf.pkl')

print("\n💾 Model berhasil disimpan sebagai 'model_c45_hoax.pkl'")
print("✅ Tahap P6.1 Selesai!")