# 🛡️ Sistem Deteksi Berita Hoax (Indonesian Fake News Detection)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

> **Tugas Besar Mata Kuliah Kecerdasan Buatan (Artificial Intelligence)** > Program Studi Teknik Informatika - Universitas Muhammadiyah Cirebon

## 📋 Tentang Proyek
Proyek ini bertujuan untuk membangun sistem cerdas yang mampu mengklasifikasikan artikel berita bahasa Indonesia menjadi dua kategori: **HOAX** atau **FAKTA**. Sistem ini dikembangkan menggunakan algoritma *Machine Learning* **Decision Tree (C4.5)** yang dikenal memiliki kemampuan interpretasi yang baik (*Explainable AI*).

Aplikasi ini dilengkapi dengan antarmuka berbasis Web (GUI) menggunakan **Streamlit**, sehingga pengguna dapat memasukkan judul atau isi berita dan mendapatkan hasil prediksi secara *real-time*.

## 👥 Tim Pengembang
Proyek ini disusun oleh Kelompok:

| Nama | NIM | Peran Utama |
| :--- | :--- | :--- |
| **Wawan Siswanto** | [251511008] | **Project Leader & Programmer** (Implementasi Kode & Setup) |
| **Ahmadi** | [NIM Ahmadi] | **Data Analyst** (Teori Dasar, Perhitungan Manual, Testing) |
| **Agus Haerul Rizal** | [NIM Agus] | **Technical Writer** (Penyusunan Laporan & Visualisasi) |

## 🚀 Fitur Utama
* **Preprocessing Otomatis:** Membersihkan teks dari simbol, URL, angka, dan *stopwords* bahasa Indonesia.
* **TF-IDF Vectorization:** Mengubah teks menjadi representasi vektor numerik.
* **Klasifikasi C4.5:** Menggunakan *Entropy* dan *Information Gain* untuk menentukan validitas berita.
* **Confidence Score:** Menampilkan tingkat keyakinan model (persentase) terhadap prediksinya.
* **Visualisasi Word Cloud:** (Opsional) Melihat kata-kata dominan pada dataset.

## 📂 Struktur Folder
Berikut adalah struktur direktori proyek ini agar mudah dipahami:

```bash
├── data/
│   ├── raw/                   # Dataset mentah (CSV terpisah)
│   └── processed/             # Dataset bersih (dataset_final_gabungan.csv)
├── model/
│   ├── model_c45.pkl          # Model Decision Tree yang sudah dilatih
│   └── tfidf_vectorizer.pkl   # Model TF-IDF yang sudah disimpan
├── notebooks/                 # File Jupyter Notebook (Analisis Data)
├── src/
│   ├── preprocessing.py       # Modul pembersihan teks
│   └── training_c45.py            # Script untuk melatih ulang model
├── app_gui.py                     # File Utama Aplikasi Streamlit
├── requirements.txt           # Daftar library yang dibutuhkan
└── README.md                  # Dokumentasi Proyek
```

## 🛠️ Instalasi & Cara Menjalankan
Pastikan Anda sudah menginstal Python dan Git di komputer Anda.

### 1. Clone Repositori
```bash
git clone https://github.com/vuelo14/tugas_besar_c4.5.git
cd tugas_besar_c4.5
```
### 2. Buat Virtual Environment (Disarankan)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Library
```bash
pip install -r requirements.txt
```
### 4. Jalankan Aplikasi
```bash
streamlit run app_gui.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`

## 📊 Dataset & Performa Model
**Dataset yang digunakan merupakan gabungan dari beberapa sumber terpercaya:**

* Hoax: Diambil dari repositori TurnBackHoax.id (MAFINDO).

* Fakta: Diambil dari portal berita Antara, Detik, dan Kompas.

* Sumber : `https://www.kaggle.com/datasets/mochamadabdulazis/deteksi-berita-hoaks-indo-dataset?resource=download`

**Statistik Dataset:**

* Total Data: 20.000+ Baris

* Ratio Kelas: ~50% Hoax : ~50% Fakta (Balanced)

**Hasil Evaluasi (Sementara):**

* Akurasi: 96.89%

* Precision: 99.0%

* Recall: 95.0%

## 📸 Screenshots


Dibuat dengan ❤️ oleh Wawan Siswanto - 2026