import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ==========================================
# 1. LOAD DATA
# ==========================================
nama_file = 'dataset_final_gabungan.csv'
try:
    df = pd.read_csv(nama_file)
    print(f"✅ Data dimuat: {len(df)} baris.")
except FileNotFoundError:
    print("❌ File tidak ditemukan. Pastikan nama file 'dataset_final_gabungan.csv' sudah benar.")
    exit()

# Tentukan nama kolom teks dan label (sesuaikan jika beda)
# Cek dengan print(df.columns) jika error
kolom_teks = 'content' if 'content' in df.columns else df.columns[1] # Mencoba menebak kolom ke-2
kolom_label = 'label'

# ==========================================
# 2. STATISTIK DESKRIPTIF (Untuk Laporan)
# ==========================================
print("\n--- STATISTIK DATASET ---")
print(df[kolom_label].value_counts())
print(f"\nTotal Data Kosong:\n{df.isnull().sum()}")

# Menghitung panjang kata per berita (Text Length)
df['jumlah_kata'] = df[kolom_teks].astype(str).apply(lambda x: len(x.split()))

print("\n--- Statistik Panjang Kata ---")
print(df.groupby(kolom_label)['jumlah_kata'].describe())

# ==========================================
# 3. VISUALISASI (3 Plot Wajib)
# ==========================================

# PLOT 1: Distribusi Kelas (Hoax vs Fakta)
plt.figure(figsize=(6, 5))
sns.countplot(x=kolom_label, data=df, palette='viridis')
plt.title('Plot 1: Perbandingan Jumlah Berita Hoax (1) vs Fakta (0)')
plt.xlabel('Label (0=Fakta, 1=Hoax)')
plt.ylabel('Jumlah')
plt.savefig('visualisasi_1_distribusi_kelas.png') # Simpan gambar otomatis
plt.show()

# PLOT 2: Histogram Panjang Kata (Apakah Hoax lebih pendek/panjang?)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='jumlah_kata', hue=kolom_label, kde=True, bins=50, palette='Set2')
plt.title('Plot 2: Sebaran Panjang Kata (Word Count) per Kategori')
plt.xlabel('Jumlah Kata dalam Berita')
plt.xlim(0, 1000) # Membatasi tampilan sumbu X agar grafik terbaca (zoom in)
plt.savefig('visualisasi_2_panjang_kata.png')
plt.show()

# PLOT 3: Word Cloud (Kata apa yang sering muncul?)
# Memisahkan teks Hoax dan Fakta
text_hoax = " ".join(df[df[kolom_label] == 1][kolom_teks].astype(str).tolist())
text_fakta = " ".join(df[df[kolom_label] == 0][kolom_teks].astype(str).tolist())

# Bikin Word Cloud Hoax
wc_hoax = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(text_hoax)
plt.figure(figsize=(10, 5))
plt.imshow(wc_hoax, interpolation='bilinear')
plt.axis('off')
plt.title('Plot 3A: Kata Sering Muncul di Berita HOAX')
plt.savefig('visualisasi_3a_wordcloud_hoax.png')
plt.show()

# Bikin Word Cloud Fakta
wc_fakta = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(text_fakta)
plt.figure(figsize=(10, 5))
plt.imshow(wc_fakta, interpolation='bilinear')
plt.axis('off')
plt.title('Plot 3B: Kata Sering Muncul di Berita FAKTA')
plt.savefig('visualisasi_3b_wordcloud_fakta.png')
plt.show()

print("\n🎉 Selesai! Cek folder Anda, gambar visualisasi sudah tersimpan.")