# Laporan Proyek 1 Machine Learning - Fiqih Alfito

## Domain Proyek

Diabetes adalah penyakit kronis yang ditandai dengan kadar gula darah yang tinggi. Ini adalah masalah kesehatan global yang signifikan, dengan jutaan orang yang terkena diabetes di seluruh dunia. Penanganan diabetes memerlukan pengawasan ketat terhadap kadar glukosa darah serta pengelolaan gaya hidup yang sehat. Dalam upaya untuk mengatasi dan mengelola diabetes, teknologi *machine learning* telah menjadi alat yang sangat berguna dalam menganalisis dan memprediksi risiko diabetes serta membantu dalam pengambilan keputusan klinis.

Pengembangan model *machine learning* yang dapat memprediksi diabetes berdasarkan data klinis individu telah menjadi fokus penelitian yang signifikan. Tujuan utama dari pengembangan model ini adalah untuk membangun sistem yang dapat mengklasifikasikan dengan akurat apakah seseorang mengalami diabetes atau tidak.

Penyakit diabetes dapat mengakibatkan komplikasi penyakit, yang tentunya sangat berbahaya terhadap penderita diabetes. Oleh karena itu, sangat diperlukanya suatu teknologi yang dapat mendeteksi penyakit diabetes dengan tingkat analisis yang akurat, sehingga penyakit diabetes dapat ditangani lebih awal untuk mengurangi jumlah penderita, kecacatan, dan kematian. Karena efek diabetes dapat menyebabkan komplikasi bahkan kematian, dibutukan model prediksi untuk mengklasifikasikan seseorang mengidap penyakit diabetes untuk mengetahui seseorang mengidap penyakit diabetes atau tidak secara cepat.[1]

## Business Understanding

### Problem Statements

Berdasarkan kondisi yang telah diuraikan sebelumnya, penulis akan mengembangkan sebuah sistem prediksi diabetes untuk menjawab permasalahan berikut.

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh untuk mengetahui seseorang yang mengalami diabetes?
- Apakah seseorang sedang mengalami diabetes atau tidak dengan karakteristik atau fitur tertentu? 

### Goals

Untuk  menjawab pertanyaan tersebut, Anda akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:

- Mengetahui fitur yang paling berkorelasi atau berpengaruh terhadap status diabetes seseorang.
- Membuat model *machine learning* yang dapat memprediksi seseorang mengalami diabetes atau tidak dengan seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution statements

- Untuk mengetahui fitur mana yang paling berkorelasi dalam mengetahui seseorang yang sedang mengalami diabetes yaitu dengan metode *Bivariate analysis* yang mana digunakan untuk membandingkan fitur diabetes dengan fitur lainnya.
- untuk mendapat hasil yang akurat, penelitian perlu menggunakan beberapa algoritma. Berhubung data yang digunakan bersifat klasifikasi, setiap model akan diukur dengan metrik ``accuracy_score()``. Skala ``accuracy_score()`` berada di rentang 1 sampai 0 yang mana 1 menunjukkan sangat akurat dan 0 sangat tidak akurat. Model yang menunjukan akurasi yang paling tinggi adalah model yang akurat dalam memprediksi diabetes.

## Data Understanding

Data yang digunakan dalam proyek ini adalah [**Dataset Prediksi Diabetes**](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset). Dataset prediksi diabetes adalah kumpulan data medis dan demografis dari 100.000 pasien, beserta status diabetesnya (positif atau negatif). Data tersebut mencakup fitur seperti *age*, *gender*, *body mass index (BMI)*, *hypertension*, *heart disease*, *smoking history*, *HbA1c level*, dan *blood glucose level*. Kumpulan data ini dapat digunakan untuk membangun model *machine learning* untuk memprediksi diabetes pada pasien berdasarkan riwayat medis dan informasi demografis mereka. 

### Variabel-variabel pada Dataset Prediksi Diabetes adalah sebagai berikut:

- Age: Usia merupakan faktor penting dalam memprediksi risiko diabetes. Seiring bertambahnya usia individu, risiko mereka terkena diabetes meningkat. Hal ini sebagian disebabkan oleh faktor-faktor seperti berkurangnya aktivitas fisik, perubahan kadar hormon, dan kemungkinan lebih tinggi untuk mengembangkan kondisi kesehatan lain yang dapat menyebabkan diabetes.

- Gender: Jenis kelamin dapat berperan dalam risiko diabetes, meskipun efeknya dapat bervariasi. Misalnya, wanita dengan riwayat diabetes gestasional (diabetes saat hamil) memiliki risiko lebih tinggi terkena diabetes tipe 2 di kemudian hari. Selain itu, beberapa penelitian menunjukkan bahwa pria mungkin memiliki risiko diabetes yang sedikit lebih tinggi dibandingkan wanita.

- Body Mass Index (BMI): BMI adalah ukuran lemak tubuh berdasarkan tinggi dan berat badan seseorang. Ini umumnya digunakan sebagai indikator status berat badan secara keseluruhan dan dapat membantu dalam memprediksi risiko diabetes. BMI yang lebih tinggi dikaitkan dengan kemungkinan lebih besar terkena diabetes tipe 2. Kelebihan lemak tubuh, terutama di sekitar pinggang, dapat menyebabkan resistensi insulin dan merusak kemampuan tubuh untuk mengatur kadar gula darah.

- Hypertension: Hipertensi, atau tekanan darah tinggi, adalah kondisi yang sering terjadi bersamaan dengan diabetes. Kedua kondisi tersebut memiliki faktor risiko yang sama dan dapat berkontribusi pada perkembangan satu sama lain. Memiliki hipertensi meningkatkan risiko terkena diabetes tipe 2 dan sebaliknya. Kedua kondisi tersebut dapat berdampak buruk pada kesehatan jantung.

- Heart Disease: Penyakit jantung, termasuk kondisi seperti penyakit arteri koroner dan gagal jantung, dikaitkan dengan peningkatan risiko diabetes. Hubungan antara penyakit jantung dan diabetes bersifat dua arah, artinya memiliki satu kondisi meningkatkan risiko berkembangnya kondisi lainnya. Ini karena mereka memiliki banyak faktor risiko yang sama, seperti obesitas, tekanan darah tinggi, dan kolesterol tinggi.

- Smoking History: Merokok merupakan faktor risiko diabetes yang dapat dimodifikasi. Merokok sigaret diketahui meningkatkan risiko diabetes tipe 2. Merokok dapat menyebabkan resistensi insulin dan merusak metabolisme glukosa. Berhenti merokok dapat secara signifikan mengurangi risiko terkena diabetes dan komplikasinya.

- HbA1c Level: HbA1c (hemoglobin terglikasi) adalah ukuran rata-rata kadar glukosa darah selama 2-3 bulan terakhir. Ini memberikan informasi tentang kontrol gula darah jangka panjang. Tingkat HbA1c yang lebih tinggi menunjukkan kontrol glikemik yang lebih buruk dan berhubungan dengan peningkatan risiko diabetes dan komplikasinya.

- Blood Glucose Level: Tingkat glukosa darah mengacu pada jumlah glukosa (gula) yang ada dalam darah pada waktu tertentu. Peningkatan kadar glukosa darah, terutama dalam keadaan puasa atau setelah mengonsumsi karbohidrat, dapat mengindikasikan gangguan regulasi glukosa dan meningkatkan risiko diabetes. Pemantauan rutin kadar glukosa darah penting dalam diagnosis dan pengelolaan diabetes


Terdapat beberapa tahapan dalam memahami dataset tersebut, yaitu:

1. Periksa duplikasi

    Kehadiran data duplikasi dalam set pelatihan dapat menghasilkan evaluasi yang tidak akurat, karena model sudah terbiasa dengan data yang sama berulang-ulang.

    library Pandas menyediakan fungsi untuk menghapus baris duplikat yaitu : 
    > `pd.drop_duplicates()`
    
    Setelah data duplikasi dihapus, jumlah data tersisa 96146.

2. periksa *missing value*

    Jika missing value tidak dikelola dengan benar, ini dapat menyebabkan kesalahan dalam analisis data, termasuk perhitungan statistik dan visualisasi. Hasil yang tidak akurat atau bias dapat mengarah pada kesimpulan yang salah atau interpretasi yang tidak benar.

    Untuk mengetahui apakah suatu fitur memiliki nilai kosong atau tidak, berikut fungsi yang digunakan :
    > `pd.isnull().sum()`
    
    karena tidak ada nilai yang kosong, maka jumlah data tetap 96146.

3. *Univariate Analysis*

    *Univariate analysis*, atau analisis univariat, adalah metode analisis statistik yang digunakan untuk memahami dan menganalisis satu fitur pada satu waktu.     Tujuan utama dari analisis univariat adalah untuk mendapatkan wawasan dan pemahaman yang mendalam tentang suatu fitur secara terpisah. Pada kasus ini digunakan diagram batang dan histogram untuk menganalisis univariat.

    analisis univariate dibagi menjadi dua yaitu analisis fitur kategorial dan fitur numerik. fitur kategorial disini terdiri dari fitur ``'gender', 'hypertension', 'heart_disease', 'smoking_history', 'diabetes'``. Untuk fitur numerik terdiri dari fitur ``'age', 'bmi', 'HbA1c_level', 'blood_glucose_level'``.

    Dari hasil analisis, maka diperoleh kesimpulan sebagai berikut:

    - Fitur `gender`. 
    
        ![Screenshot_20230611_215522](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/557902dc-33f3-4001-b90c-40cc33e67459)
    
        Gambar 1. Analisis univariat fitur `gender`
    
        Tabel 1. Hasil analisis univariat fitur `gender`
    
        |               | jumlah sampel | persentase |
        | ------------- | ------------- | ---------- |
        | Female        | 56161         | 58.4       |
        | Male          | 39967         | 41.6       |
    
        Berdasarkan gambar 1 diatas. Dari seluruh pasien, jumlah perempuan lebih banyak dibandingan dengan laki-laki. Berdasarkan tabel 1, jumlah perempuan sebanyak 58.4% dan 41.6% adalah laki-laki.
    
    - Fitur `hypertension`. 
    
        ![Screenshot_20230611_222157](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/de6551a5-1355-4e16-a5b3-8d0b74287e2a)

        Gambar 2. Analisis univariat fitur `hypertension`
    
        Tabel 2. Hasil analisis univariat fitur `hypertension`
    
        |   | jumlah sampel | persentase |
        | - | ------------- | ---------- |
        | 0 | 88667         | 92.2       |
        | 1 | 7461          | 7.8        |
    
        Berdasarkan gambar 2 diatas. Dari seluruh pasien, jumlah pasien dengan kondisi hipertensi lebih sedikit dibandingan dengan yang tidak hipertensi. Berdasarkan tabel 2, jumlah pasien dengan kondisi hipertensi sebanyak 7.8% dan 92.2% adalah pasien yang tidak memiliki kondisi hipertensi.
    
    
    - Fitur `heart_disease`.
        
        ![Screenshot_20230611_223409](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/c2329299-5f65-4fbc-b44c-133658c51814)
        
        Gambar 3. Analisis univariat fitur `heart_disease`
        
        Tabel 3. Hasil analisis univariat fitur `heart_disease`
        
        |   | jumlah sampel | persentase |
        | - | ------------- | ---------- |
        | 0 | 92205         | 95.9       |
        | 1 | 3923          | 4.1        |
        
        Berdasarkan gambar 3 diatas. Dari seluruh pasien, jumlah pasien yang mempunyai penyakit jantung lebih sedikit dibandingan dengan yang tidak mempunyai penyakit jantung. Berdasarkan tabel 3, jumlah pasien yang mempunyai penyakit jantung sebanyak 4.1% dan 95.9% adalah pasien yang tidak mempunyai penyakit jantung.
   
    - Fitur `smoking_history`. Dari seluruh pasien, 70% pasien tidak pernah merokok.
            
        ![Screenshot_20230611_223734](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/da91d2b9-f277-4262-b8e5-cb491124842b)
        
        Gambar 4. Analisis univariat fitur `smoking_history`
        
        Tabel 4. Hasil analisis univariat fitur `smoking_history`
        
        |              | jumlah sampel  | persentase |
        | ------------ | -------------  | ---------- |
        | never        |         34395  |       35.8 |
        | No Info      |         32881  |       34.2 |
        | former       |          9299  |        9.7 |
        | current      |          9197  |        9.6 |
        | not current  |          6359  |        6.6 |
        | ever         |          3997  |        4.2 |
        
        Berdasarkan gambar 4 diatas. Dari seluruh pasien, pasien yang tidak pernah merokok lebih dominan. Berdasarkan tabel 4, Sebesar 70% pasien tidak pernah merokok. 
        
    - Fitur `diabetes`
            
        ![Screenshot_20230611_224419](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/8f1d6944-cfbb-46a5-baa8-466d78fb26ef)
        
        Gambar 5. Analisis univariat fitur `diabetes`
        
        Tabel 5. Hasil analisis univariat fitur `diabetes`
        
        |   | jumlah sampel | persentase |
        | - | ------------- | ---------- |
        | 0 | 87646         | 91.2       |
        | 1 | 8482          | 8.8        |
               
        
        Berdasarkan gambar 5 diatas. Dari seluruh pasien, jumlah pasien yang menderita diabetes lebih sedikit dari pada pasien yang tidak menderita diabetes. Berdasarkan tabel 5, pasien yang tidak menderita diabetes sebesar 91.2% dan 8.8% adalah pasien yang menderita diabetes.
     
    
4. *Bivariate Analysis*

    *Bivariate analysis*, atau analisis bivariat, adalah metode analisis statistik yang digunakan untuk memahami hubungan atau interaksi antara dua fitur secara simultan. Dalam analisis bivariat, fokus diberikan pada bagaimana fitur satu mempengaruhi fitur lainnya.

    Dari hasil analisis, maka diperoleh kesimpulan sebagai berikut:

    - *Gender* dan *Diabetes*

        ![Screenshot_20230611_225240](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/b9924089-04bd-46cd-964e-c1393b57e376)

        Gambar 6. Analisis fitur `gender` dan `diabetes` 

        Berdasarkan pada gambar 6 di atas, jumlah perempuan dan laki-laki yang mengalami diabetes hampir setara, perempuan sedikit lebih banyak dari laki-laki.
    
    - *Age* dan *Diabetes*
    
        ![Screenshot_20230611_230147](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/4c7d5368-0139-4be6-b71d-8441ac888326)
        
        Gambar 7. Analisis fitur `age` dan `diabetes`

        Berdasarkan pada gambar 7 di atas, kecenderungan untuk terkena diabetes perlahan naik dimulai pada usia 30 dan diabetes lebih banyak terjadi pada usia 60 keatas.
    
    - *Hypertension* dan *Diabetes*

        ![Screenshot_20230611_230219](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/31cd6489-2dfd-44b8-94f7-31e5abbe1cad)
        
        Gambar 8. Analisis fitur `hypertension` dan `diabetes`

        Berdasarkan pada gambar 8 di atas, pasien yang tidak mengalami hipertensi lebih banyak mengalami diabetes dari pada pasien yang mengalami hipertensi.
    
    - *Heart Disease* dan *Diabetes*

        ![Screenshot_20230611_230238](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/f330ef15-a17d-46bd-9327-51839301a16f)
        
        Gambar 9. Analisis fitur `heart_disease` dan `diabetes`

        Berdasarkan pada gambar 9 di atas, pasien yang tidak memiliki penyakit jantung lebih banyak mengalami diabetes dari pada pasien yang memiliki penyakit jantung.
    
    - *Smoking History* dan *Diabetes*

        ![Screenshot_20230611_230259](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/f9af2752-0075-4f34-88ea-2c39d8e26ea4)
        
        Gambar 10. Analisis fitur `smoking_history` dan `diabetes`

        Berdasarkan pada gambar 10, pasien yang mengalami diabetes sebagian besar yang tidak pernah merokok dan mantan perokok.

    - *Body Mass Index* dan *Diabetes*

        ![Screenshot_20230611_230317](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/2a6a0631-e759-4e35-840a-17283f8557ab)

        Gambar 11. Analisis fitur `bmi` dan `diabetes`

        Berdasarkan pada gambar 11 di atas, dengan meningkatnya BMI, peluang untuk terkena diabetes akan meningkat.
    
    - *HbA1c Level* dan *Diabetes*

        ![Screenshot_20230611_230336](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/19ce8e7e-5b8b-4739-accc-0f89e2346167)

        Gambar 12. Analisis fitur `HbA1c_Level` dan `diabetes`

        Berdasarkan gambar 12 di atas, dengan meningkat HbA1c Level, peluang untuk terkena diabetes akan meningkat. rata-rata pasien yang mengalami diabetes ketika HbA1c Level sebesar 6 keatas.
    
    - *Blood Glucose Level* dan *Diabetes*

        ![Screenshot_20230611_230356](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/5879e18c-67fd-4bc5-abac-c57604425770)

        Gambar 13. Analisis fitur `blood_glucose_level` dan `diabetes`
    
        Berdasarkan gambar 13 di atas, dengan meningkat Blood Glucose Level, peluang untuk terkena diabetes akan meningkat. rata-rata pasien yang mengalami diabetes ketika Blood Glucose Level sebesar 150 keatas.


5. *Multivariate Analysis*

    Analisis multivariat merujuk pada kumpulan teknik statistik yang digunakan untuk menganalisis dan memahami hubungan antara beberapa fitur secara simultan. Berbeda dengan analisis univariat (yang berfokus pada satu fitur) atau analisis bivariat (yang mempelajari hubungan antara dua fitur), analisis multivariat mempertimbangkan interaksi dan ketergantungan antara tiga atau lebih fitur. Dalam kasus ini, analisis multivariate digunakan seberapa besar korelasi semua fitur terhadap fitur diabetes agar fitur yang memiliki korelasi yang kecil dapat disingkirkan.
    
    Untuk evaluasi skor korelasi antar fitur khususnya fitur numerik, peneliti menggunakan fungsi `corr()` lalu skor yang didapat akan ditampilkan dalam diagram heatmap agar mudah dibaca.
    
    ![Screenshot_20230611_231927](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/a27338eb-6a61-4993-850a-7dbebe00e9f0)
    
    Gambar 14. Matriks korelasi pada fitur numerik

    Berdasarkan gambar 14 di atas, dari hasil analisis, fitur ``heart_disease`` memiliki korelasi paling kecil karena skornya mendekati angka 0. Koefisien korelasi berkisar antara -1 dan +1. Ia mengukur kekuatan hubungan antara dua variabel serta arahnya (positif atau negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilainya ke 1 atau -1, korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0, korelasinya semakin lemah. Selanjutnya fitur tersebut di-*drop*. Maka sekarang fitur yang tersisa adalah `gender`, `age`, `hypertension`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`, dan `diabetes`. 

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

Berikut persiapan data yang dilakukan yaitu:

1. Encoding Fitur Kategori

    Setelah melakukan analisis, tersisa beberapa fitur kategorikal dan numerik. fitur kategorikal perlu diubah ke fitur numerik agar model dapat memperlakukan setiap kategori secara terpisah dan tidak memberikan nilai peringkat atau urutan yang tidak relevan. Dengan cara ini, kita dapat menggunakan data kategori dalam model *machine learning* atau analisis data dengan lebih efektif.

    Fitur `gender` dan `smoking_history` adalah fitur kategorikal. Maka perlu diubah ke fitur numerik dengan fungsi pandas `get_dummies()`. Setelah berubah menjadi fitur numerik lalu drop fitur kategorikal sebelumnya. Alhasil semua fitur adalah fitur numerik.

    *One-hot encoding* perlu dilakukan karena banyak algoritma pembelajaran mesin dan analisis data hanya dapat bekerja dengan data numerik. Mereka tidak dapat langsung memproses atau memahami data kategorikal. Jika data kategorikal dianggap sebagai fitur numerik, ini menghilangkan informasi tentang perbedaan antara kategori. Misalnya, jika ada kolom "jenis kelamin" dengan nilai "pria" dan "wanita", dan kita memberikan angka 0 dan 1 masing-masing, ini tidak mencerminkan perbedaan yang sebenarnya antara kategori tersebut. Ini dapat memberikan kesan bahwa ada hubungan ordinal di antara kategori tersebut, meskipun sebenarnya tidak ada. Ini bisa menyesatkan algoritma pembelajaran mesin yang mengasumsikan hubungan ordinal.

2. Train-Test-Split

    Setelah semua fitur berupa numerik, kemudian lakukan *train-test-split*. *train-test-split* adalah metode yang digunakan dalam pembelajaran mesin dan statistik untuk membagi data menjadi dua subset yang saling terpisah, yaitu data pelatihan (training data) dan data pengujian (testing data).

    Selanjutnya pilih fitur target atau fitur yang akan dijadikan prediksi yaitu fitur `diabetes` lalu tandai sebagai variabel "**y**". Fitur-fitur selain fitur target ditandai dengan variabel "**X**".

    *Sklearn* sudah menyiapakan fungsi untuk membagi data latih dan data uji yaitu ``train_test_split()``. Lalu lakukan pemisahan dengan fungsi tersebut, maka data tersebut akan terpisah dengan ditandai nama variabel *train* dan *test*. 
    
    Berhubung ukuran dataset cukup besar, maka rasio pembagian data latih dan data uji yaitu 90:10. Umumnya rasio yang digunakan 80:20 jika dataset memiliki sedikit data. Total data sebelumnya sebesar 96128. Setelah dibagi dengan rasio 90:10, maka total data latih sebesar 86515 dan data uji sebesar 9613.

    Dengan membagi data menjadi subset pelatihan dan pengujian, kita dapat menghindari penilaian yang terlalu optimis dan mendapatkan perkiraan yang lebih realistis tentang seberapa baik model akan berperforma pada data yang tidak pernah dilihat sebelumnya.

3. Standarisasi

    Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn.
 
    StandardScaler adalah salah satu teknik standarisasi yang umum digunakan dalam machine learning. Teknik ini berguna untuk mengubah distribusi data menjadi memiliki mean (rerata) 0 dan standar deviasi (simpangan baku) 1. Dengan menggunakan StandardScaler, setiap fitur akan diperlakukan secara independen dan diskala ulang sehingga memiliki skala yang seragam.

    fitur-fitur yang akan dilakukan standarisasi yaitu ``'age', 'bmi', 'HbA1c_level', 'blood_glucose_level'`` agar fitur-fitur tersebut memiliki skala yang seragam.

    Jika fitur-fitur memiliki skala yang berbeda-beda, interpretasi hasil model dapat menjadi sulit. Misalnya, sulit untuk membandingkan bobot atau koefisien antar fitur jika mereka memiliki skala yang berbeda. Juga, interpretasi pengaruh setiap fitur dalam model dapat menjadi lebih rumit jika skala tidak seragam.


## Modeling

Sebelum memulai *modeling*, siapkan terlebih dahulu *dataframe* untuk menyimpan hasil evaluasi model. Evaluasi model pada proyek ini menggunakan metrik `accuracy`. Dataframe terdiri dari kolom dan baris. Nama kolom terdiri dari `train_accuracy_score` dan `test_accuracy_score`. Nama baris terdiri dari nama-nama algoritma yang digunakan.

Pada tahap *modeling*, digunakan beberapa algoritma klasifikasi untuk memprediksi status diabetes pasien yaitu *K-Nearest Neighbor Classifier*, *Random Forest Classifier*, *Boosting Classifier*.

- **K-Nearest Neighbor Classifier**

    K-Nearest Neighbor (K-NN) Classifier adalah salah satu metode klasifikasi yang populer dalam machine learning. Pendekatan ini digunakan untuk mengklasifikasikan data baru berdasarkan kesamaan atau jaraknya terhadap data yang ada dalam dataset pelatihan. Dalam K-NN, kata "K" mengacu pada jumlah tetangga terdekat yang akan digunakan untuk menentukan kelas atau label data baru.

    Kelebihan K-NN Classifier:

    1. Sederhana dan mudah dipahami: Konsep K-NN relatif sederhana dan mudah dipahami. Hal ini membuatnya cocok untuk pemula dalam machine learning.

    2. Tidak memerlukan proses pembelajaran: K-NN termasuk dalam metode pembelajaran berbasis instansi (instance-based), yang berarti tidak memerlukan proses pelatihan yang kompleks. Model K-NN secara efektif "menghafal" data pelatihan dan menggunakannya langsung saat melakukan prediksi.

    3. Toleran terhadap perubahan data: K-NN dapat dengan mudah menyesuaikan diri dengan perubahan dalam data pelatihan. Jika ada penambahan data baru, model K-NN tidak perlu dilatih ulang, melainkan hanya menambahkan data tersebut ke dataset yang ada.

    Kekurangan K-NN Classifier:

    1. Komputasi yang mahal: Ketika dataset menjadi sangat besar, K-NN dapat menjadi komputasi yang mahal. Menghitung jarak antara data baru dengan setiap titik data dalam dataset pelatihan bisa memakan waktu yang cukup lama.

    2. Sensitif terhadap data yang tidak relevan: K-NN dapat menjadi sensitif terhadap data yang tidak relevan dalam dataset. Data yang tidak relevan dapat menyebabkan distorsi dalam perhitungan jarak dan menghasilkan prediksi yang tidak akurat.

    3. Memerlukan pemrosesan data yang tepat: Sebelum menggunakan K-NN, sering kali diperlukan pemrosesan data untuk menghilangkan atribut yang tidak relevan, mengisi nilai yang hilang, atau melakukan normalisasi. Pemrosesan data yang tepat dapat membantu meningkatkan kinerja K-NN.
    
    Parameter yang digunakan :
    
    - `n_neighbors`: Parameter ini menentukan jumlah tetangga terdekat yang akan digunakan dalam proses klasifikasi. Nilai yang umumnya digunakan adalah bilangan ganjil untuk menghindari situasi kesetimbangan kelas. Jumlah tetangga yang terlalu rendah dapat menghasilkan model yang sensitif terhadap noise, sementara jumlah tetangga yang terlalu tinggi dapat menghasilkan model yang terlalu umum. Untuk proyek ini, model ini menggunakan nilai default yaitu 5. Artinya, KNN Classifier akan mempertimbangkan 5 tetangga terdekat dalam menghasilkan prediksi.

    Hasil dari pelatihan model K-NN Classifier pada data latih akan dievaluasi menggunakan metrik `accuracy_score()` dan akan disimpan pada variabel `models.loc['KNN', 'train_accuracy_score']`.

- **Random Forest Classifier**

    Random Forest Classifier adalah sebuah algoritma yang digunakan dalam machine learning untuk melakukan klasifikasi data. Algoritma ini bekerja dengan menggabungkan beberapa pohon keputusan (decision trees) yang bekerja secara independen dan menghasilkan prediksi berdasarkan mayoritas suara atau rata-rata dari prediksi masing-masing pohon.

    Berikut adalah beberapa kelebihan dari Random Forest Classifier:

    1. Akurasi yang Tinggi: Random Forest Classifier sering memberikan hasil yang sangat akurat dalam klasifikasi data. Hal ini disebabkan oleh fakta bahwa algoritma ini menggabungkan prediksi dari banyak pohon keputusan, sehingga dapat mengurangi overfitting dan mengatasi bias yang ada dalam setiap pohon individu.

    2. Toleransi terhadap Fitur Irrelevan: Algoritma ini mampu menangani dataset dengan fitur yang tidak relevan atau tidak penting. Saat membangun setiap pohon keputusan, hanya sebagian kecil dari fitur yang digunakan secara acak untuk membatasi pengaruh fitur yang kurang penting terhadap prediksi.

    3. Keandalan terhadap Noise: Random Forest Classifier dapat menangani dataset yang mengandung noise atau outlier. Ketika prediksi dibuat dengan memperhitungkan mayoritas suara dari pohon-pohon individu, dampak dari data yang salah atau noise dapat dikurangi.

    4. Kecepatan dan Skalabilitas: Algoritma ini dapat bekerja dengan cepat pada dataset besar dan memiliki kemampuan untuk memproses banyak fitur. Dengan mengoptimalkan paralelisme, Random Forest Classifier dapat dijalankan secara efisien pada sistem dengan sumber daya yang cukup.

    Namun, Random Forest Classifier juga memiliki beberapa kelemahan:

    1. Tidak Interpretatif: Hasil dari Random Forest Classifier mungkin sulit diinterpretasikan. Karena algoritma ini menggabungkan prediksi dari banyak pohon keputusan, sulit untuk menguraikan bagaimana setiap fitur berkontribusi terhadap prediksi akhir.

    2. Kompleksitas Model: Random Forest Classifier membangun banyak pohon keputusan, yang dapat menyebabkan kompleksitas model yang tinggi. Hal ini dapat menyebabkan waktu pelatihan yang lama, terutama untuk dataset yang besar, serta memerlukan sumber daya komputasi yang cukup tinggi.

    3. Overfitting pada Data dengan Fitur Lebih Banyak: Jika dataset memiliki jumlah fitur yang sangat besar dibandingkan dengan jumlah sampel, Random Forest Classifier dapat mengalami overfitting. Dalam kasus ini, algoritma cenderung terlalu menyesuaikan diri dengan data pelatihan dan kinerjanya pada data yang tidak dikenal dapat menurun.

    4. Pengaturan Parameter yang Penting: Random Forest Classifier memiliki beberapa parameter yang perlu dikonfigurasi dengan benar, seperti jumlah pohon dalam ensemble dan kedalaman maksimum setiap pohon. Pengaturan parameter yang tidak tepat dapat mempengaruhi kinerja model secara keseluruhan.

    Parameter yang digunakan :
    
    - `n_estimators`: Parameter ini menentukan jumlah pohon keputusan yang akan dibangun dalam model Random Forest. Semakin banyak pohon yang digunakan, semakin kompleks model dan waktu pelatihannya akan meningkat. Untuk proyek ini, model ini menggunakan nilai default yaitu 100. Artinya, Random Forest Classifier akan menggunakan 100 pohon keputusan dalam menghasilkan prediksi.
    - `max_depth`: Parameter ini menentukan kedalaman maksimum dari setiap pohon dalam model. Kedalaman yang lebih dalam dapat menghasilkan model yang lebih kompleks, tetapi juga dapat menyebabkan overfitting. Jika tidak ditentukan, pohon tidak memiliki batasan kedalaman dan akan terus membagi sampai semua node menjadi murni atau sampai jumlah sampel minimum untuk split tidak terpenuhi. Untuk proyek ini, model tidak menentukan parameter `max_depth`, untuk itu model akan memasang nilai default yaitu `None`.

    Hasil dari pelatihan model Random Forest Classifier pada data latih akan dievaluasi menggunakan metrik `accuracy_score()` dan akan disimpan pada variabel `models.loc['RandomForest', 'train_accuracy_score']`.

- **Boosting Classifier**

    Ada beberapa jenis pada algoritma *boosting*. Dalam penelitian ini dipakai algoritma *AdaBoostClassifier*. 
    AdaBoostClassifier adalah algoritma pembelajaran mesin yang digunakan untuk klasifikasi. Ini adalah jenis algoritma ensemble yang menggabungkan beberapa model pembelajaran mesin sederhana, yang disebut "weak learners", untuk membentuk model prediksi yang lebih kuat. Kelemahan masing-masing weak learners dikompensasi dengan memfokuskan pada sampel yang dianggap sulit untuk diklasifikasikan oleh weak learner sebelumnya.

    Berikut adalah penjelasan singkat tentang cara kerja AdaBoostClassifier:

    1. Inisialisasi bobot: Setiap sampel dalam dataset diberi bobot awal yang sama.

    2. Pelatihan weak learners: Weak learner pertama dilatih pada dataset dengan bobot awal. Mereka menghasilkan model prediksi sederhana.

    3. Evaluasi dan pembaruan bobot: Bobot dihitung ulang berdasarkan seberapa baik weak learner pertama mengklasifikasikan sampel. Sampel yang salah diklasifikasikan diberi bobot yang lebih tinggi untuk fokus pada mereka dalam langkah berikutnya.

    4. Pelatihan weak learners berikutnya: Weak learner kedua dilatih pada dataset dengan bobot yang diperbarui. Proses ini diulang untuk weak learners yang lain, dengan bobot diperbarui setiap kali.

    5. Pemilihan bobot model: Bobot setiap weak learner dihitung berdasarkan akurasi mereka. Weak learners dengan akurasi yang lebih tinggi diberi bobot yang lebih tinggi dalam pemilihan model akhir.

    6. Penggabungan model: Prediksi dari setiap weak learner dikombinasikan berdasarkan bobot mereka untuk menghasilkan prediksi final.

    Kelebihan AdaBoostClassifier:

    1. Performa yang tinggi: Dalam banyak kasus, AdaBoostClassifier menghasilkan kinerja yang lebih baik dibandingkan dengan penggunaan weak learner individu.

    2. Tidak ada kebutuhan untuk penyetelan parameter yang rumit: AdaBoostClassifier secara otomatis menyesuaikan bobot sampel dan bobot model untuk menghasilkan model akhir yang kuat.

    3. Mampu menangani data dengan fitur kompleks: AdaBoostClassifier dapat mengatasi data yang memiliki fitur kompleks dan tidak linear.

    Kekurangan AdaBoostClassifier:

    1. Rentan terhadap overfitting: Jika jumlah weak learners terlalu besar atau dataset memiliki noise yang tinggi, AdaBoostClassifier dapat cenderung overfitting pada data pelatihan.

    2. Sensitif terhadap outliers: Keberadaan outliers dalam dataset dapat mempengaruhi performa AdaBoostClassifier karena bobotnya diperbarui berdasarkan kesalahan klasifikasi.

    3. Sensitif terhadap data yang tidak seimbang: Jika dataset memiliki kelas yang tidak seimbang secara signifikan, AdaBoostClassifier cenderung memberikan bobot yang lebih tinggi pada kelas mayoritas dan mungkin memiliki kinerja yang buruk pada kelas minoritas.

    Parameter yang digunakan :
    
    - `n_estimators`: Parameter ini mengindikasikan jumlah maksimum dari estimator (weak learner) yang akan digunakan dalam ensemble. Semakin besar nilai n_estimators, semakin kompleks model akan menjadi. Namun, nilai yang terlalu besar juga dapat menyebabkan overfitting. Pada proyek ini, model ini menggunakan nilai default yaitu 50. Dalam AdaBoostClassifier, ensemble terdiri dari sejumlah estimator yang dilatih secara berurutan. Estimator adalah model pembelajaran mesin sederhana yang digunakan dalam AdaBoost untuk mempelajari aturan prediksi yang lebih kompleks. Pada setiap iterasi, estimator baru ditambahkan ke ensemble dengan tujuan untuk meningkatkan performa prediksi keseluruhan. Setiap estimator fokus pada sampel yang sulit diprediksi oleh estimator sebelumnya dengan memberikan bobot lebih besar pada sampel-sampel tersebut.
    - `learning_rate`: Parameter ini mengontrol kontribusi setiap estimator terhadap keseluruhan model. Nilai learning_rate yang lebih kecil akan menghasilkan model yang lebih konservatif dengan meningkatkan stabilitas, tetapi juga memerlukan lebih banyak estimator untuk mencapai performa yang setara. Untuk proyek ini, model ini menetapkan nilai `learning_rate` defaultnya adalah 1.

    Hasil dari pelatihan model Boosting Classifier pada data latih akan dievaluasi menggunakan metrik `accuracy_score()` dan akan disimpan pada variabel `models.loc['Boosting', 'train_accuracy_score']`.

## Evaluation

Untuk mengevaluasi model yang telah dilatih, proyek ini menggunakan metrik `accuracy`. Metrik akurasi adalah ukuran yang digunakan untuk mengevaluasi sejauh mana model machine learning dapat memprediksi dengan benar kelas target pada data yang diberikan.

### Metrik Accuracy

Metrik akurasi adalah salah satu metrik evaluasi yang digunakan dalam model machine learning untuk mengukur seberapa akurat model dalam melakukan prediksi kelas target. Akurasi mengukur sejauh mana model berhasil memprediksi dengan benar kelas target pada data yang diberikan.

Untuk menghitung akurasi, langkah pertama adalah membandingkan setiap prediksi yang dilakukan oleh model dengan nilai sebenarnya dari kelas target. Kemudian, jumlah prediksi yang benar dibagi dengan total jumlah prediksi, dan hasilnya dikalikan dengan 100 untuk mendapatkan persentase akurasi.

Berikut rumus matematika untuk menghitung akurasi:

Akurasi = (Jumlah prediksi yang benar / Total jumlah prediksi) * 100

Misalnya, jika terdapat 100 data yang diprediksi oleh model, dan model benar-benar memprediksi dengan benar 80 dari 100 data tersebut, maka akurasi model adalah 80%.

Akurasi adalah metrik evaluasi yang penting, terutama dalam masalah klasifikasi di mana kelas target yang diinginkan adalah biner (dua kelas).

Pada proyek ini, nilai accuracy dari setiap model akan digunakan untuk membandingkan model mana yang terbaik. Rentang pada nilai akurasi yaitu dari 0 sampai 1. Nilai accuracy model yang paling mendekati 1 akan menjadi model yang terbaik.

Hasil dari evaluasi model menggunakan metrik akurasi sebagai berikut:

Tabel 1. Nilai accuracy data latih dan data uji

|                  | **train_accuracy_score** | **test_accuracy_score** |
| ---------------- | ------------------------ | ----------------------- |
| **KNN**          | 0.9691                   | 0.9597                  |
| **RandomForest** | 0.9989                   | 0.9675                  |
| **Boosting**     | 0.9709                   | 0.9707                  |

Berdasarkan tabel di atas, dapat diambil beberapa kesimpulan:

1. Model KNN memiliki tingkat akurasi yang cukup tinggi baik pada data latihan maupun pada data uji. Dalam hal ini, akurasi model pada data latihan adalah 0,9691, sedangkan pada data uji adalah 0,9597. Hal ini menunjukkan bahwa model KNN berhasil menghasilkan prediksi yang akurat pada data baru yang belum pernah dilihat sebelumnya.

2. Model Random Forest memiliki tingkat akurasi yang sangat tinggi pada data latihan dengan nilai 0,9989, yang menunjukkan bahwa model ini hampir sempurna dalam mempelajari pola dari data latihan. Namun, meskipun tingkat akurasi pada data uji (0,9675) tidak sebaik pada data latihan, model ini masih tetap berhasil memberikan hasil yang cukup baik pada data baru.

3. Model Boosting juga menunjukkan tingkat akurasi yang tinggi pada data latihan (0,9709) dan data uji (0,9707). Hal ini menunjukkan bahwa model Boosting mampu memberikan prediksi yang konsisten pada kedua set data tersebut.

## Kesimpulan

Dalam kesimpulan, semua model yang dievaluasi dalam tabel tersebut memiliki tingkat akurasi yang relatif tinggi. Namun, model Random Forest memiliki tingkat akurasi tertinggi pada data latihan, sedangkan model Boosting memiliki tingkat akurasi yang cukup tinggi dan konsisten pada kedua set data. Pilihan model terbaik tergantung pada konteks dan kebutuhan spesifik dari masalah yang dihadapi. Sehingga model terbaik yang dipilih yaitu model *Boosting* karena dapat menunjukan hasil yang konsisten pada evaluasi data uji dan data latih.

## Referensi

[1]     A. Supandi, A. Faqih, and F. Basysyar, "PREDIKSI PENYAKIT DIABETES MENGGUNAKAN MACHINE LEARNING DENGAN ALGORITMA NAIVE BAYES", *JS*, vol. 10, no. 2, pp. 146 - 152, Aug. 2022. [Available](https://ejournal.indobarunasional.ac.id/index.php/jursima/article/view/396)






