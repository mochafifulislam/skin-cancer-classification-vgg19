# 🩺 Klasifikasi Citra Lesi Kulit Menggunakan VGG19 Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Proyek ini bertujuan untuk mengklasifikasikan 7 jenis lesi kulit (kanker dan non-kanker) dari dataset **HAM10000** menggunakan pendekatan *Transfer Learning* dengan arsitektur **VGG19**.

---

## 📌 Daftar Isi
- [Latar Belakang](#-latar-belakang)
- [Dataset](#-dataset)
- [Arsitektur Model](#-arsitektur-model)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi & Penggunaan](#-instalasi--penggunaan)
- [Hasil & Evaluasi](#-hasil--evaluasi)
- [Lisensi](#-lisensi)
- [Kontak](#-kontak)

---

## 🔬 Latar Belakang
Deteksi dini kanker kulit seperti Melanoma sangat krusial untuk meningkatkan tingkat kelangsungan hidup pasien. Penggunaan *Deep Learning* dalam citra medis (dermatoskopi) dapat membantu dokter dan spesialis kulit dalam melakukan diagnosis awal secara cepat dan akurat.

Proyek ini memanfaatkan model **VGG19** yang dimodifikasi (*Fine-Tuning*) untuk mengenali pola visual pada citra lesi kulit.

---

## 📊 Dataset: HAM10000
Dataset yang digunakan adalah **HAM10000** (*Human Against Skin Cancer*), yang mencakup 10.015 citra dermatoskopi berlabel.

### 7 Kategori Lesi Kulit:
1. **`nv`** - Melanocytic nevi (Tahi lalat jinak)
2. **`mel`** - Melanoma (Ganas)
3. **`bkl`** - Benign keratosis-like lesions
4. **`bcc`** - Basal cell carcinoma (Ganas)
5. **`akiec`** - Actinic keratoses / Bowen's disease
6. **`vasc`** - Vascular lesions
7. **`df`** - Dermatofibroma

---

## 🏗️ Arsitektur Model
- **Base Model**: VGG19 (pre-trained di ImageNet, *weights frozen* pada tahap awal).
- **Custom Classifier Head**:
  - `GlobalAveragePooling2D` / `Flatten`
  - `BatchNormalization`
  - `Dense` (256/512 unit, aktivasi ReLU)
  - `Dropout` (0.3 - 0.5) untuk mencegah overfitting
  - `Dense` (7 unit, aktivasi Softmax)
- **Optimizer**: Adam (Learning Rate: 1e-4)
- **Loss Function**: Categorical Crossentropy

---

## ⚙️ Instalasi & Penggunaan

### 1. Clone Repositori
```bash
git clone [https://github.com/mochafifulislam/skin-cancer-classification-vgg19.git](https://github.com/mochafifulislam/skin-cancer-classification-vgg19.git)
cd skin-cancer-classification-vgg19
