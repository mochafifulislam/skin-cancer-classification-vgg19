# 📂 Petunjuk Dataset HAM10000

Folder ini digunakan untuk menyimpan dataset **HAM10000** (*Human Against Skin Cancer*) yang dibutuhkan untuk pelatihan dan evaluasi model.

> ⚠️ **Catatan:** File dataset asli berukuran besar dan diabaikan oleh Git (`.gitignore`). Anda perlu mengunduhnya terlebih dahulu sebelum menjalankan kode pelatihan.

---

## ⬇️ Cara Mengunduh Dataset

### Opsi 1: Unduh Otomatis (Menggunakan `kagglehub`)
Jika Anda menjalankan script `src/train.py` atau Notebook `notebooks/skin_cancer_vgg19.ipynb`, dataset akan diunduh secara otomatis dari Kaggle ke dalam cache komputer Anda.

### Opsi 2: Unduh Manual dari Kaggle
1. Kunjungi halaman dataset [Kaggle - Skin Cancer MNIST HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Unduh file dataset zip.
3. Ekstrak seluruh isi file zip ke dalam folder `data/` ini.

---

## 📁 Struktur Folder `data/` yang Diharapkan

Setelah diunduh dan ekstrak selesai, pastikan struktur folder di dalam `data/` terlihat seperti berikut:

```text
data/
├── README.md
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
│   ├── ISIC_0024306.jpg
│   ├── ISIC_0024307.jpg
│   └── ...
└── HAM10000_images_part_2/
    ├── ISIC_0029306.jpg
    ├── ISIC_0029307.jpg
    └── ...
