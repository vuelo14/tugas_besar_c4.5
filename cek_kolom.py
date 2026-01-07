import pandas as pd

try:
    # Coba baca file yang dianggap bermasalah
    df = pd.read_csv('dataset_bersih_siap_training.csv')
    
    print("✅ File ditemukan!")
    print(f"Jumlah Baris: {len(df)}")
    print("\n--- DAFTAR KOLOM YANG ADA ---")
    print(list(df.columns))
    
    if 'text_clean' in df.columns:
        print("\n✅ Kolom 'text_clean' ADA. Seharusnya training_c45.py bisa jalan.")
    else:
        print("\n❌ Kolom 'text_clean' TIDAK ADA.")
        print("👉 SOLUSI: Anda WAJIB menjalankan ulang script 'preprocessing.py' dulu!")

except FileNotFoundError:
    print("❌ File 'dataset_bersih_siap_training.csv' TIDAK DITEMUKAN.")
    print("👉 Pastikan Anda sudah menjalankan preprocessing.py")