import pandas as pd

print("Sedang memuat data...")

# 1. BACA DATASET (Load Files)
# Pastikan nama file sesuai persis dengan yang ada di folder
df_hoax   = pd.read_csv('Cleaned_TurnBackHoax_v3.csv')
df_antara = pd.read_csv('Cleaned_Antaranews_v1.csv')
df_detik  = pd.read_csv('Cleaned_Detik_v2.csv')
df_kompas = pd.read_csv('Cleaned_Kompas_v2.csv')

# 2. BERI LABEL (Labeling)
# Kita buat kolom baru bernama 'label'
# 1 = Hoax, 0 = Fakta

df_hoax['label'] = 1
df_antara['label'] = 0
df_detik['label'] = 0
df_kompas['label'] = 0

print(f"Jumlah Data Hoax: {len(df_hoax)}")
print(f"Jumlah Data Antara: {len(df_antara)}")
print(f"Jumlah Data Detik: {len(df_detik)}")
print(f"Jumlah Data Kompas: {len(df_kompas)}")

# 3. GABUNGKAN DATA (Concatenate)
# Kita ambil semua data fakta, lalu gabung dengan data hoax
# Opsional: Kita batasi jumlah data fakta agar seimbang (balanced) dengan hoax
# Tapi untuk tahap awal, kita gabung semuanya dulu saja.

df_gabungan = pd.concat([df_hoax, df_antara, df_detik, df_kompas], ignore_index=True)

# 4. ACAK DATA (Shuffle)
# Penting agar urutannya tidak Hoax dulu baru Fakta semua
df_gabungan = df_gabungan.sample(frac=1).reset_index(drop=True)

# 5. CEK HASIL
print("\n--- Data Berhasil Digabung ---")
print(df_gabungan.info())
print("\n--- Contoh 5 Baris Data ---")
print(df_gabungan.head())

# 6. SIMPAN FILE BARU
# File ini yang nanti akan dipakai seterusnya untuk Preprocessing & Training
nama_file_baru = 'dataset_final_gabungan.csv'
df_gabungan.to_csv(nama_file_baru, index=False)

print(f"\n✅ Sukses! File baru tersimpan sebagai: {nama_file_baru}")