# 🎙️ Dashboard Pengenal Angka Suara Bahasa Indonesia

Dashboard Streamlit untuk mengenali **deretan angka yang diucapkan** (mis. *"sembilan satu kosong tiga dua"* → `91032`) memakai model yang dilatih dengan notebook `ADL_DASHBOARD_4_SPEECH_RECOGNITION_ROBUST.ipynb` pada dataset INDspeech_DIGIT_CDSR.

## Fitur
| Fitur | Keterangan |
|---|---|
| 🎤 Rekam langsung | Rekam dari mikrofon browser, hasil langsung muncul |
| 🔇 Penjernihan suara | Mode Otomatis / Selalu / Mati: buang dengung, derau latar, klik, dan hening sebelum dikenali; kualitas rekaman (SNR) ditampilkan dan audio hasil penjernihan bisa didengar |
| 🧠 CRNN-CTC Robust | Model deep learning yang dilatih dengan augmentasi derau (ESC-50, babble, gema, simulasi mic HP) |
| 📁 Unggah file | WAV, MP3, M4A, OGG, FLAC |
| 🧪 Contoh dataset | Rekaman uji asli beserta jawaban benarnya untuk demo |
| ⭐ Pilih metode + rekomendasi | 6 metode; metode terbaik (skor gabungan WER bersih + bising, diuji statistik) ditandai ⭐ dan dipilih otomatis |
| 🟢🟠 Keyakinan per digit | Digit yang kurang yakin ditandai oranye agar petugas mengecek ulang |
| 🏦 Skenario layanan | Cek panjang nomor (antrean, rekening, HP, tiket) |
| 🧾 Mode bertahap | Nomor panjang diucapkan per kelompok, lalu digabung otomatis |
| 🔬 Bandingkan semua metode | Satu rekaman dijalankan di semua metode sekaligus |
| 📊 Perbandingan metode | Akurasi + selang kepercayaan 95%, uji signifikansi, WER per panjang nomor, confusion matrix |
| 📝 Riwayat | Riwayat sesi bisa diunduh sebagai CSV |

## Struktur folder
```
streamlit_dashboard_cdsr/
├── app.py                  # aplikasi dashboard
├── cdsr_core.py            # modul inti (SAMA dengan yang ditulis notebook)
├── requirements.txt        # pustaka Python
├── packages.txt            # pustaka sistem (ffmpeg) untuk Streamlit Cloud
├── .streamlit/config.toml  # tema warna
└── artifacts/              # ← isi dengan hasil notebook
```

## Langkah 1 — Hasilkan artefak model (Colab)
1. Buka notebook di Colab, pilih **Runtime → Change runtime type → T4 GPU**.
2. Jalankan semua sel dari atas sampai bawah. Ekstraksi fitur pertama memakan beberapa menit; sesi berikutnya memakai cache.
3. Notebook membuat `dashboard_artifacts.zip` di Drive: `MyDrive/Tugas kuliah/ADL/`.
4. Unduh, lalu **ekstrak isinya ke folder `artifacts/`** sehingga berisi:
   `config.json`, `metrics.json`, `test_predictions.csv`, `cdsr_core.py`, `models/`, `samples/`.

> Bila notebook diubah, salin juga `artifacts/cdsr_core.py` menimpa `cdsr_core.py` di folder utama agar kodenya tetap sama.

## Langkah 2 — Jalankan di komputer sendiri
```bash
cd streamlit_dashboard_cdsr
pip install -r requirements.txt
streamlit run app.py
```
Buka `http://localhost:8501`. Perekaman mikrofon hanya diizinkan browser di `localhost` atau `https`.

## Langkah 3 — (Opsional) Online di Streamlit Community Cloud
1. Unggah folder ini (termasuk `artifacts/`) ke repositori GitHub.
2. Di [share.streamlit.io](https://share.streamlit.io) pilih repositori dan file `app.py`.
3. Model sudah dikompresi; bila ada file di `artifacts/models/` lebih dari 100 MB, gunakan Git LFS.

## ⚠️ Versi pustaka harus sama
Model scikit-learn/XGBoost hanya aman dibuka dengan versi yang sama seperti saat dilatih. Lihat `artifacts/config.json` bagian `versions`, lalu kunci versinya di `requirements.txt`, contoh:
```
scikit-learn==1.6.1
xgboost==2.1.4
```
Dashboard menampilkan peringatan di sidebar bila versinya berbeda.

## Pemecahan masalah
| Gejala | Solusi |
|---|---|
| "Folder artifacts/ belum berisi model" | Ekstrak `dashboard_artifacts.zip` ke `artifacts/` |
| Metode CNN-BiLSTM-CTC tidak muncul | PyTorch belum terpasang: `pip install torch` |
| MP3/M4A gagal dibaca | Pasang ffmpeg, atau unggah WAV |
| Hasil rekaman mikrofon kurang akurat | Pakai metode **CRNN-CTC Robust** dan mode penjernihan **Otomatis**; dekatkan mikrofon dan beri jeda singkat antar-angka |
| Metode CRNN-CTC Robust tidak muncul | Jalankan notebook versi 4 sampai selesai lalu salin ulang rtifacts/ (harus ada models/crnn_robust.pt) |

## Batasan penggunaan
- Jangan gunakan untuk PIN, kata sandi, CVV, atau OTP.
- Selalu konfirmasi ulang nomor ke nasabah. Ini prototipe akademik, bukan sistem produksi transaksi keuangan.
- Rekaman diproses di memori dan tidak disimpan ke disk oleh aplikasi.

**Dataset:** Sakti, S., Hutagaol, P., Arman, A. A., & Nakamura, S. (2004). *Indonesian Speech Recognition for Hearing and Speaking Impaired People.* Proc. INTERSPEECH–ICSLP, 1037–1040. Lisensi CC-BY-NC-SA-4.0.
