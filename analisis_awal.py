import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# BAGIAN 1: LOAD DATASET (Tugas P5.1)
# ==========================================

# Nama Dataset
# nama_file = 'dataset_hoax_v1.csv' 
nama_file = 'dataset_final_gabungan.csv' 

try:
    # Membaca file CSV
    df = pd.read_csv(nama_file)
    print("✅ Berhasil memuat dataset!")
    
    # Menampilkan 5 baris pertama data untuk pengecekan
    print("\n--- 5 Data Pertama ---")
    print(df.head())

except FileNotFoundError:
    print("❌ File tidak ditemukan! Pastikan nama file sesuai dan ada di folder yang sama.")

# ==========================================
# BAGIAN 2: PRELIMINARY ANALYSIS (Tugas P5.2)
# ==========================================

if 'df' in locals():
    print("\n--- Informasi Dataset ---")
    df.info()

    # 1. Cek Apakah ada data kosong (Missing Values)
    print("\n--- Cek Data Kosong ---")
    print(df.isnull().sum())

    # 2. Cek Sebaran Label (Berapa Hoax vs Fakta?)
    # Asumsi nama kolom labelnya adalah 'label'
    target_column = 'label'  
    
    if target_column in df.columns:
        jumlah_label = df[target_column].value_counts()
        print(f"\n--- Jumlah Data per Kategori ({target_column}) ---")
        print(jumlah_label)

        # 3. Visualisasi Grafik Batang (Bar Chart)
        plt.figure(figsize=(6, 4))
        sns.countplot(x=target_column, data=df, palette='viridis')
        plt.title('Perbandingan Jumlah Berita Hoax vs Fakta')
        plt.xlabel('Kategori')
        plt.ylabel('Jumlah Berita')
        plt.show()
        
        print("\n✅ Grafik berhasil ditampilkan. Simpan gambar ini untuk Laporan Pendahuluan.")
    else:
        print(f"\n⚠️ Kolom '{target_column}' tidak ditemukan. Cek nama kolom dengan print(df.columns)")