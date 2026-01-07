import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ==============================================================================
# MODUL P7.2: EXPERIMENTAL RESULTS (ANALISIS PERFORMA)
# Output: Gambar 'confusion_matrix.png' dan Teks Laporan
# ==============================================================================

print("🚀 MEMULAI ANALISIS PERFORMA...")

# 1. LOAD DATA & MODEL
# Kita harus membagi data persis sama dengan saat training (random_state=42)
# agar data testing-nya valid (bukan data yang sudah dihafal model).
try:
    df = pd.read_csv('dataset_bersih_siap_training.csv')
    df['text_clean'] = df['text_clean'].fillna('')
    
    clf = joblib.load('model_c45_hoax.pkl')
    tfidf = joblib.load('vectorizer_tfidf.pkl')
except Exception as e:
    print(f"❌ Error loading: {e}")
    exit()

# 2. PERSIAPKAN DATA TESTING (20%)
X = tfidf.transform(df['text_clean']).toarray()
y = df['label']

# PENTING: random_state harus SAMA dengan training_c45.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. PREDIKSI
print("⏳ Sedang melakukan prediksi pada data testing...")
y_pred = clf.predict(X_test)

# 4. MEMBUAT CONFUSION MATRIX (HEATMAP)
print("📊 Membuat Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prediksi Fakta', 'Prediksi Hoax'],
            yticklabels=['Asli Fakta', 'Asli Hoax'])
plt.title('Confusion Matrix: Model C4.5 Deteksi Hoax')
plt.ylabel('Label Sebenarnya')
plt.xlabel('Label Prediksi AI')
plt.savefig('confusion_matrix.png')
plt.show()

# 5. CETAK LAPORAN STATISTIK
print("\n" + "="*40)
print("HASIL EKSPERIMEN (Untuk Copy-Paste ke Laporan)")
print("="*40)
print(classification_report(y_test, y_pred, target_names=['Fakta', 'Hoax']))
print(f"Total Data Testing: {len(y_test)}")
print(f"Benar Prediksi (Fakta & Hoax): {cm[0,0] + cm[1,1]}")
print(f"Salah Prediksi (Error): {cm[0,1] + cm[1,0]}")
print("="*40)
print("✅ Gambar tersimpan: 'confusion_matrix.png'")