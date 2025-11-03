# Proyek Prediksi Risiko Gagal Bayar (Credit Default Risk) - Home Credit

Proyek ini adalah studi kasus *end-to-end* (Task 5) untuk Home Credit Indonesia. Tujuannya adalah membangun model *machine learning* yang mampu memprediksi probabilitas seorang klien akan gagal bayar (default) pada pinjamannya.

Fokus utama proyek ini adalah menangani **dataset tidak seimbang** (hanya **8.07%** klien yang gagal bayar) dan melakukan **feature engineering** secara ekstensif dari berbagai sumber data untuk memaksimalkan daya prediksi model.

**File dataset**: https://drive.google.com/drive/folders/1kjNaMcro4qKVbkzbSxax8s-9CLgKRqwb?usp=sharing

**Model Final:** LightGBM (LGBM) Tuned
**Skor AUC-ROC Final:** **0.7717**

## 🎯 1. Permasalahan Bisnis & Metodologi

* **Permasalahan:** Mengidentifikasi klien berisiko tinggi (TARGET=1) dari mayoritas klien lancar (TARGET=0) untuk meminimalkan kerugian kredit.
* **Metrik Evaluasi:** Karena data tidak seimbang, **Akurasi** bukanlah metrik yang tepat. Metrik utama yang digunakan adalah:
    * **AUC-ROC:** Sebagai metrik utama untuk mengukur kemampuan model dalam membedakan kedua kelas.
    * **Recall:** Untuk memastikan kita menangkap sebanyak mungkin klien yang benar-benar berisiko.
    * **Precision:** Untuk meminimalkan jumlah klien baik yang salah ditandai (False Positive).

## 🛠️ 2. Alur Kerja Proyek

Proyek ini mengikuti alur kerja *data science* yang lengkap, mulai dari pembersihan data hingga *feature engineering* dan *tuning*.

### A. Data Preprocessing
1.  **Data Cleaning:** Menangani 67 kolom dengan *missing values*. 37 kolom (missing > 50%) dihapus, dan 30 kolom sisanya di-imputasi menggunakan Median (untuk numerik) dan Modus (untuk kategorikal).
2.  **Encoding:** **One-Hot Encoding** (`pd.get_dummies`) diterapkan pada 14 kolom kategorikal.
3.  **Scaling:** **StandardScaler** digunakan untuk menormalkan semua fitur, memastikan semua fitur setara secara matematis untuk model Regresi Logistik.

### B. Pemodelan & Peningkatan Iteratif
Model dievaluasi dan ditingkatkan secara bertahap:

1.  **Model Baseline (Regresi Logistik):** Sesuai instruksi, model ini dibuat sebagai patokan awal dengan `class_weight='balanced'`.
    * **AUC: 0.7454**

2.  **Model Menengah (LightGBM):** Beralih ke model *tree-based* yang lebih kuat, juga dengan `class_weight='balanced'`.
    * **AUC: 0.7547**

3.  **Hyperparameter Tuning (LGBM):** Menggunakan `RandomizedSearchCV` (dijalankan di GPU Colab) untuk menemukan parameter terbaik untuk LGBM.
    * **AUC: 0.7553**

### C. Feature Engineering (Peningkatan Kunci)
Ini adalah inti dari peningkatan model, di mana data dari file eksternal diagregasi dan digabungkan.

4.  **+ Fitur `previous_application.csv` (13 Fitur):** Menambahkan fitur seperti "total penolakan sebelumnya" & "rata-rata pinjaman sebelumnya".
    * **AUC: 0.7609**

5.  **+ Fitur `bureau.csv` & `bureau_balance.csv` (37 Fitur):** Menambahkan fitur seperti "jumlah kredit aktif di bank lain" & "riwayat keterlambatan bayar eksternal".
    * **AUC: 0.7651**

6.  **+ Fitur `installments_payments.csv` (27 Fitur):** Menambahkan fitur perilaku paling kuat, seperti "rata-rata hari keterlambatan bayar cicilan" & "total kekurangan bayar".
    * **AUC FINAL: 0.7717**

## 🏆 3. Hasil Model Final

Model juara adalah **LightGBM** yang telah di-*tuning* dan dilatih pada **275 fitur** gabungan.

| Metrik | Skor | Interpretasi |
| :--- | :--- | :--- |
| **AUC-ROC** | **0.7717** | Model ini 77.17% lebih baik daripada tebakan acak dalam membedakan klien. |
| **Recall** | **0.6926** | Berhasil mengidentifikasi **69.3%** dari semua klien yang akan gagal bayar. |
| **Precision**| **0.1757** | Dari semua klien yang diprediksi berisiko, **17.6%** adalah prediksi yang benar. |

<img width="1004" height="744" alt="image" src="https://github.com/user-attachments/assets/d3a07e18-c022-4075-8de5-51c7d3d3b25f" />

## 📈 4. Rekomendasi Bisnis

Model dengan AUC **0.7717** ini siap untuk diimplementasikan sebagai **Sistem Skor Risiko (Risk Scoring System)**.

3.  **Manajemen Kerugian:** Dengan *recall* **69.3%**, model ini secara proaktif mengidentifikasi lebih dari dua pertiga potensi kerugian, memungkinkan perusahaan untuk menolak atau menyesuaikan syarat pinjaman (misal: menaikkan DP, memperpendek tenor) untuk aplikasi berisiko tersebut.

## 📁 5. Struktur Repositori

* `Model Home Credit Indonesia.ipynb`: *Notebook* Google Colab yang berisi semua alur kerja analisis, *preprocessing*, *feature engineering*, dan *modelling*.
* `README.md`: Penjelasan proyek ini.
