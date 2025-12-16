# Streamlit App — Klasifikasi Tingkat Kebisingan

Aplikasi Streamlit ini memuat model TensorFlow Lite `best_cnn_model_weighted_augmented.tflite` untuk mengklasifikasikan tingkat kebisingan ruangan berdasarkan log-Mel spectrogram (n_mels=128, n_fft=1024, hop_length=256, sr=22.05 kHz) sesuai preprocessing pada notebook pelatihan.

## Struktur Folder

```
streamlit_app/
├── app.py
├── models/
│   └── best_cnn_model_weighted_augmented.tflite
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── audio_processing.py
    └── inference.py
```

## Menjalankan Secara Lokal

1. **Buat dan aktifkan environment** (opsional tetapi direkomendasikan).
2. Instal dependensi:
   ```powershell
   pip install -r requirements.txt
   ```
3. Jalankan Streamlit:
   ```powershell
   streamlit run app.py
   ```
4. Buka URL yang ditampilkan (biasanya `http://localhost:8501`).

## Fitur Utama

- Dua pilihan input: unggah file WAV atau rekam langsung via komponen `audio-recorder-streamlit`.
- Preprocessing identik dengan notebook: resampling ke 22.05 kHz, log-Mel spectrogram (128 band), padding/truncation ke 638 frame.
- Inferensi menggunakan TensorFlow Lite interpreter dengan label `Sedang`, `Tenang`, `Tinggi`.
- Visualisasi probabilitas dan log-Mel spectrogram dari audio yang diproses.

## Catatan Deploy

- Pastikan file model `.tflite` ikut disertakan saat deploy (mis. ke Streamlit Community Cloud). Path relatif sudah mengarah ke `models/best_cnn_model_weighted_augmented.tflite`.
- Jika ingin mengecilkan ukuran dependensi, Anda dapat mengganti TensorFlow penuh dengan `tflite-runtime` sesuai platform dan memperbarui `requirements.txt`.
- Format audio yang direkomendasikan adalah WAV mono agar konversi berjalan konsisten. Untuk format lain, konversi ke WAV terlebih dahulu.
