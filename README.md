# 🔊 Klasifikasi Tingkat Kebisingan Ruangan Menggunakan CNN  
### Streamlit-Based Deep Learning Application

Aplikasi ini merupakan implementasi **model Convolutional Neural Network (CNN)** untuk melakukan **klasifikasi tingkat kebisingan ruangan** berbasis data audio. Aplikasi dikembangkan menggunakan **Streamlit** sebagai antarmuka interaktif untuk memudahkan pengujian model secara langsung melalui web.

Penelitian ini bertujuan untuk **mengevaluasi potensi CNN sebagai pendekatan awal** dalam pengembangan **sistem deteksi kebisingan kampus secara otomatis**, khususnya sebagai solusi pendukung monitoring kenyamanan lingkungan belajar.

---

## 📌 Latar Belakang

Tingkat kebisingan merupakan salah satu faktor penting yang memengaruhi **kenyamanan dan konsentrasi di lingkungan kampus**. Metode pengukuran kebisingan secara manual memiliki keterbatasan dalam hal efisiensi dan keberlanjutan pengamatan.

Oleh karena itu, pendekatan berbasis **deep learning** digunakan untuk mengklasifikasikan tingkat kebisingan secara otomatis. **Convolutional Neural Network (CNN)** dipilih karena kemampuannya dalam mengekstraksi fitur spasial dari representasi sinyal audio seperti **Mel-Spectrogram**, yang umum digunakan dalam klasifikasi audio.

---

## 🎯 Tujuan Penelitian

- Mengimplementasikan model **CNN** untuk klasifikasi tingkat kebisingan ruangan
- Mengevaluasi performa CNN sebagai pendekatan awal sistem deteksi kebisingan
- Mengembangkan aplikasi berbasis **Streamlit** untuk pengujian model
- Mendukung pengembangan sistem monitoring kebisingan lingkungan kampus

---

## 🧠 Kelas Kebisingan

Model mengklasifikasikan tingkat kebisingan ke dalam beberapa kategori, antara lain:

- 🔵 **Rendah (Low Noise)**
- 🟡 **Sedang (Medium Noise)**
- 🔴 **Tinggi (High Noise)**

Kategori dapat disesuaikan dengan standar kebisingan dan dataset yang digunakan.

---

## 🚀 Fitur Aplikasi

Aplikasi Streamlit ini memiliki beberapa fitur utama, yaitu:

- 📁 Upload file audio berformat `.wav`
- 🔄 Preprocessing audio otomatis
- 📊 Ekstraksi fitur menggunakan **Mel-Spectrogram**
- 🧠 Prediksi tingkat kebisingan menggunakan model CNN
- 📈 Visualisasi hasil ekstraksi fitur
- 🧾 Tampilan hasil klasifikasi secara real-time

---

## 📂 Struktur Direktori

```text
streamlit_app/
├── app.py                     # Main Streamlit application
├── model/
│   └── cnn_model.h5           # Model CNN terlatih
├── utils/
│   ├── preprocessing.py       # Audio preprocessing
│   └── feature_extraction.py  # Feature extraction (Mel-Spectrogram)
├── assets/                    # Dataset contoh / file pendukung
├── requirements.txt           # Dependencies
└── README.md
```
---
## ⚙️ Teknologi yang Digunakan

Python \
Streamlit \
TensorFlow / Keras \
Librosa \
NumPy \
Matplotlib \
Scikit-learn

---
## 🛠️ Cara Menjalankan Aplikasi
1️⃣ Clone Repository
```
git clone https://github.com/username/nama-repo.git
cd streamlit_app
```

2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
3️⃣ Jalankan Aplikasi
```
streamlit run app.py
```

Aplikasi dapat diakses melalui browser pada:
```
http://localhost:8501
```
---
## 📊 Evaluasi Model

Evaluasi model CNN dilakukan menggunakan beberapa metrik, antara lain:

Accuracy

Precision

Recall

Confusion Matrix

Hasil evaluasi menunjukkan bahwa CNN memiliki potensi yang baik sebagai baseline model dalam pengembangan sistem klasifikasi kebisingan ruangan.

---
## 🌐 Deployment

Aplikasi ini dapat dideploy menggunakan beberapa platform, seperti:

Streamlit Community Cloud

Render

Hugging Face Spaces

Pastikan file requirements.txt dan app.py berada pada root aplikasi saat proses deployment.

---

## 📌 Kontribusi Akademik

Proyek ini dikembangkan sebagai bagian dari Tugas Besar / Proyek Mata Kuliah Deep Learning dan dapat dikembangkan lebih lanjut untuk:

Integrasi dengan sensor suara (IoT)

Sistem deteksi kebisingan secara real-time

Monitoring lingkungan kampus berbasis data

---
## 🛡️ Lisensi

Proyek ini menggunakan lisensi MIT License.

---
## 👥 Penulis
Kayla Amanda Sukma \
NABIILAH PUTRI KARNAIA  
MEIRA LISTYANINGRUM 
