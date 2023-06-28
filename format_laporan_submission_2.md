# Laporan Proyek Machine Learning - Fiqih Alfito

## Project Overview

Sistem rekomendasi adalah suatu mekanisme yang dapat memberikan suatu informasi atau rekomendasi sesuai dengan kesukaan user berdasarkan informasi yang diperoleh dari user. Oleh karena itu, diperlukan model rekomendasi yang tepat agar rekomendasi yang diberikan oleh sistem sesuai dengan kesukaan user, serta mempermudah user mengambil keputusan dalam  enentukan item (film) yang akan dipilih.

Sistem rekomendasi film menjadi penting dalam konteks industri hiburan dan digital, di mana jumlah konten film dan TV yang tersedia sangat luas dan terus bertambah. Dalam lingkungan seperti ini, pengguna seringkali menghadapi kesulitan dalam menavigasi melalui pilihan yang melimpah. Sistem rekomendasi film bertujuan untuk membantu pengguna menemukan konten yang paling relevan dan menarik bagi mereka, sehingga memperbaiki pengalaman konsumen dan meningkatkan retensi pengguna. Dengan memberikan rekomendasi yang akurat, sistem ini juga membantu meningkatkan kepuasan pengguna dan memfasilitasi penemuan konten baru yang mungkin tidak akan ditemukan secara konvensional[1].

## Business Understanding

### Problem Statements

Berdasarkan kondisi yang telah diuraikan sebelumnya, penulis akan mengembangkan sebuah sistem prediksi diabetes untuk menjawab permasalahan berikut.

Menjelaskan pernyataan masalah:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Approach” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution approach (algoritma atau pendekatan sistem rekomendasi).

## Data Understanding

Data yang digunakan pada proyek ini adalah [Dataset Movie Recommendation](https://www.kaggle.com/datasets/uttam94/recommendation). Dataset ini memiliki dua file csv yang terdiri dari movies.csv dan ratings.csv. movies.csv berisi data-data *movie* yang mempunyai fitur ``movieId, title, genres`` dan mempunyai 9742 baris. ratings.csv berisi data-data user yang telah melakukan *rating* pada *movie* yang telah ditontonnya, file ini terdiri dari fitur ``userId, movieId, rating, timestamp`` dan mempunyai 100836 baris. Kumpulan data ini dapat digunakan untuk membangun model *machine learning* untuk memberikan rekomendasi film pada *user* berdasarkan data-data pada film dan perilaku *user* lainnya. 

Variabel-variabel pada *Dataset movie recommendation* adalah sebagai berikut:

1. Data Movies
    
    - moviesId : merupakan id dari sebuah film.
    - title : merupakan judul dari sebuah film.
    - genres : merupakan genre atau kategori dari sebuah film. satu film bisa terdiri dari satu atau lebih genre.

2. Data Ratings

    - userId : merupakan id dari *user*. id ini bisa digunakan untuk mengetahui jumlah *user* yang telah melakukan *rating*.
    - movieId : merupakan id movie yang telah ditonton oleh *user*.
    - rating : merupakan angka penilaian dari sebuah film yang telah ditonton oleh *user*.
    - timestamp : merupakan waktu ketika *user* telah melakukan *rating*.

Terdapat beberapa tahapan dalam memahami dataset tersebut, yaitu:

1. Periksa jumlah data dan type data

    Terdapat fungsi dari pandas yaitu ``df.info()``. digunakan untuk melihat informasi dari sebuah *dataframe* Pandas seperti jumlah data dan type data dari setiap fitur. Berikut adalah info yang dihasilkan dari setiap *dataframe*.

    - Data movie

    ```
    RangeIndex: 9742 entries, 0 to 9741
    Data columns (total 3 columns):
    #   Column   Non-Null Count  Dtype 
    --  ------   --------------  ----- 
    0   movieId  9742 non-null   int64 
    1   title    9742 non-null   object
    2   genres   9742 non-null   object
    dtypes: int64(1), object(2)
    memory usage: 228.5+ KB
    ```

    - Data rating

    ```
    RangeIndex: 100836 entries, 0 to 100835
    Data columns (total 4 columns):
    #   Column     Non-Null Count   Dtype  
    --  ------     --------------   -----  
    0   userId     100836 non-null  int64  
    1   movieId    100836 non-null  int64  
    2   rating     100836 non-null  float64
    3   timestamp  100836 non-null  int64  
    dtypes: float64(1), int64(3)
    memory usage: 3.1 MB
    ```


2. Filter jumlah data

    ``unique()`` digunakan untuk mendapatkan nilai unik dari sekumpulan data. kumpulan data unik ini bisa digunakan untuk mengetahui jumlah data sebenarnya.

    ```py
    print('Banyak movie dalam ratings :', len(ratings.movieId.unique()))
    print('Banyak user dalam ratings :', len(ratings.userId.unique()))
    ```

    ```
    Banyak movie dalam ratings : 9724
    Banyak user dalam ratings : 610
    ```

3. Mengetahui jumlah rating

    Untuk mengetahui jumlah rating dari sebuah film yang telah dinilai dari beberapa *user*. Pandas menyediakan fungsi ``groupby().sum()``. ``groupby()`` diisi dengan parameter fitur yang ingin dikelompokkan. fungsi ``sum()`` akan menjumlahkan fitur-fitur yang berisi angka, termasuk selain fitur `rating`. berikut tampilan kodenya:
    ```
    all_movies.groupby('movieId').sum()
    ``` 
    maka tampil tabel berisi *movie* yang telah dikelompokkan beserta jumlah *rating*.

    | movieId | userId | rating |    timestamp    |
    |---------|--------|--------|-----------------|
    |    1    | 65904  | 843.0  | 242914455479    |
    |    2    | 36251  | 377.5  | 124938583322    |
    |    3    | 14747  | 169.5  | 52265734386     |
    |    4    |  1539  |  16.5  | 6290052048      |
    |    5    | 14679  | 150.5  | 48640552594     |
    |  ...    |  ...   |  ...   |      ...        |
    | 193581  |  184   |  4.0   | 1537109082      |
    | 193583  |  184   |  3.5   | 1537109545      |
    | 193585  |  184   |  3.5   | 1537109805      |
    | 193587  |  184   |  3.5   | 1537110021      |
    | 193609  |  331   |  4.0   | 1537157606      |



## Data Preparation

Berikut persiapan data yang dilakukan yaitu:

1. Mengatasi missing value

    Data yang digunakan harus diperiksa dahulu, apakah data tersebut terdapat nilai kosong atau tidak. nilai yang kosong dapat menimbulkan bias terhadap hasil pelatihan model *machine learning*. Berikut fungsi yang digunakan untuk mengetahui nilai kosong.

    ```
    all_movies.isnull().sum()
    ```
    
    maka akan tampil sebagai berikut:
    
    ```
    userId       0
    movieId      0
    rating       0
    timestamp    0
    title        0
    genres       0
    dtype: int64
    ```

    Dari data diatas bahwa semua fitur tidak memiliki nilai yang kosong.

2. Drop fitur yang tidak digunakan

    Periksa fitur yang tidak terlalu berpengaruh terhadap hasil rating. Dari fitur yang ada, terdapat fitur ``timestamp``. fitur ini tidak digunakan, oleh karena itu fitur ini harus dihilangkan, jika digunakan maka akan menimbulkan bias pada model. cara menghilangkannya yaitu dengan fungsi ``drop()`` dengan parameter nama kolom yang ingin dihapus. Berikut kodenya:
    
    ```
    ratings = ratings.drop(columns=['timestamp'])
    ```

    maka fitur ``timestamp`` akan hilang

    | userId | movieId | rating |
    |--------|---------|--------|
    |   1    |    1    |  4.0   |
    |   1    |    3    |  4.0   |
    |   1    |    6    |  4.0   |
    |   1    |   47    |  5.0   |
    |   1    |   50    |  5.0   |
    |  ...   |   ...   |  ...   |
    |  610   | 166534  |  4.0   |
    |  610   | 168248  |  5.0   |
    |  610   | 168250  |  5.0   |
    |  610   | 168252  |  5.0   |
    |  610   | 170875  |  3.0   |

3. Pisahkan movie yang tidak ada genre.

    Karena genre digunakan sebagai fitur rekomendasi, hilangkan movie yang tidak memiliki genre. Periksa nama-nama genre yang tersedia.

    ```py
    genre_list = movies.genres.str.split("|").tolist()
    genre = list(set(itertools.chain(*genre_list)))
    genre
    ```

    ```py
    ['Crime', 'Comedy', 'Western', 'Animation', 'Musical', 'Thriller', 'Mystery', 'Romance', 'War', 'Action', 'Fantasy', 'Horror', '(no genres listed)', 'Children', 'Drama', 'IMAX', 'Adventure', 'Sci-Fi', 'Documentary', 'Film-Noir']
    ```

    Dari data diatas, terdapat nama genre `'(no genres listed)'`, maka data ini harus dihilangkan, agar data ini tidak dianggap sebagai nama genre. movie yang tidak memiliki genre dapat mengakibatkan bias pada model.

    Berikut nama-nama movie yang memiliki data genre `'(no genres listed)'`.

    ```py
    movies[movies['genres'] == '(no genres listed)']
    ```

    | movieId | title                        | genres              |
    |---------|------------------------------|---------------------|
    | 165489  | Ethel & Ernest (2016)        | (no genres listed)  |
    | 166024  | Whiplash (2013)              | (no genres listed)  |
    | 167570  | The OA                       | (no genres listed)  |
    | 169034  | Lemonade (2016)              | (no genres listed)  |
    | 171495  | Cosmos                       | (no genres listed)  |
    | 171631  | Maria Bamford: Old Baby      | (no genres listed)  |
    | ...     | ...                          | (no genres listed)  |


    Lakukan drop baris yang memiliki genres `'(no genres listed)'`.

    ```py
    fix_movies = movies.drop(movies[movies['genres'] == '(no genres listed)'].index)
    print("jumlah movie tidak ada genre : ", len(fix_movies[fix_movies['genres'] == '(no genres listed)']))
    ```

    ```
    jumlah movie tidak ada genre :  0
    ```



## Modeling

Pada tahap *modeling*, digunakan dua solusi rekomendasi dengan algoritma berbeda yaitu Content Based Filtering dan Collaborative Filtering.

### Content Based Filtering

Content-Based Filtering adalah metode dalam sistem rekomendasi yang menggunakan karakteristik atau konten suatu item untuk merekomendasikan item lain kepada pengguna. Metode ini mengandalkan informasi yang terkandung dalam item-item yang sudah diketahui kesukaan atau preferensi pengguna untuk melakukan rekomendasi. 

Prinsip utama dari Content-Based Filtering adalah mencocokkan preferensi pengguna dengan fitur atau konten dari item-item yang ada. Fitur-fitur ini dapat berupa atribut-atribut seperti judul, genre, aktor, sutradara, atau kata kunci yang terkait dengan item tersebut. Dengan menganalisis kesesuaian fitur-fitur ini, sistem rekomendasi dapat mengidentifikasi item yang paling cocok untuk direkomendasikan kepada pengguna.

Kelebihan Content-Based Filtering:
1. Personalisasi: Metode ini dapat memberikan rekomendasi yang personal dan sesuai dengan preferensi pengguna. Dengan menganalisis karakteristik item yang disukai oleh pengguna, rekomendasi yang dihasilkan cenderung sesuai dengan preferensi individu.
2. Tidak membutuhkan data pengguna: CBF tidak memerlukan informasi pengguna selain preferensi awal yang sudah diketahui. Ini berguna jika sistem tidak memiliki akses ke data pengguna yang lengkap atau jika pengguna ingin menjaga privasi mereka.
3. Kemampuan menghadapi cold-start problem: CBF dapat bekerja dengan baik saat menghadapi cold-start problem, yaitu ketika sistem harus merekomendasikan item kepada pengguna baru atau item baru yang belum banyak diketahui.

Kekurangan Content-Based Filtering:
1. Keterbatasan variasi: Metode ini cenderung membatasi rekomendasi pada item yang memiliki fitur atau karakteristik serupa dengan item yang sudah disukai oleh pengguna. Ini dapat menyebabkan kurangnya variasi dalam rekomendasi, sehingga pengguna mungkin tidak diperkenalkan dengan item-item baru atau berbeda.
2. Tidak memperhitungkan preferensi sosial: CBF hanya mempertimbangkan preferensi individu pengguna dan tidak memperhitungkan preferensi sosial atau rekomendasi dari pengguna lain. Hal ini dapat mengabaikan kemungkinan pengaruh sosial dalam preferensi pengguna.
3. Ketergantungan pada kualitas konten: Efektivitas CBF sangat tergantung pada kualitas dan keakuratan informasi konten yang dianalisis. Jika atribut-atribut yang digunakan tidak cukup representatif atau tidak mencerminkan preferensi pengguna secara akurat, rekomendasi yang dihasilkan mungkin tidak relevan atau tepat.

Pada proyek ini, Content Based Filtering diawali dengan TF-IDF Vectorizer.

1. **TF-IDF Vectorizer**

    TF-IDF Vectorizer adalah algoritma yang mengubah teks menjadi representasi vektor numerik. Ini menggunakan konsep term frequency (frekuensi kata) dan inverse document frequency (kebalikan frekuensi dokumen). Term frequency mengukur seberapa sering kata muncul dalam suatu dokumen, sementara inverse document frequency mengukur pentingnya kata dalam seluruh dokumen. Dengan menggabungkan kedua nilai ini, TF-IDF Vectorizer menghasilkan vektor numerik yang merepresentasikan teks. Ini digunakan dalam berbagai tugas pemrosesan bahasa alami seperti klasifikasi teks dan pengambilan informasi. Dalam proyek ini, algoritma ini digunakan untuk mengambil fitur penting pada fitur ``genres``.

    pertama, inisialisasi dahulu algoritmanya.

    ```py
    # Inisialisasi TfidfVectorizer
    tf = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w-]*\w\b")
    ```

    Dalam proyek ini, TfidfVectorizer diisi dengan parameter ``token_pattern=r"(?u)\b\w[\w-]*\w\b"``. parameter tersebut bertujuan untuk mengambil data penting dengan syarat kata yang berisi *hyphenated word* tidak dipisah. Di fitur genre terdapat genre ``Sci-fi`` dan ``Film-Noir``, maka kata *hyphenated word* tersebut tidak akan dipisah menjadi ``Sci`` dan ``Fi``. Kata-kata tersebut akan dianggap sebagai satu genre yang utuh. Berikut fitur penting yang dihasilkan:

    ```
    array(['action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 'horror', 'imax', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western'], dtype=object)
    ``` 

    Kemudian melakukan fit lalu transformasikan ke bentuk matrix.

    ```py
    tfidf_matrix = tf.fit_transform(fix_movies['genres'])
    ```

    Untuk melihat tf-idf matrix, buatlah *DataFrame* dengan fitur penting yang dihasilkan sebelumnya sebagai kolom dan ``title`` *movie* sebagai baris.

    ```py
    pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=fix_movies.title
    ).sample(10, axis=1).sample(10, axis=0)
    ```

    | title                                      | thriller | children | musical | animation | fantasy | action   | war     | horror | film-noir | imax     |
    |--------------------------------------------|----------|----------|---------|-----------|---------|----------|---------|--------|-----------|----------|
    | Osama (2003)                               | 0.000000 | 0.0      | 0.0     | 0.0       | 0.0     | 0.000000 | 0.00000 | 0.0    | 0.000000  | 0.000000 |
    | The Darkest Minds (2018)                    | 0.624702 | 0.0      | 0.0     | 0.0       | 0.0     | 0.000000 | 0.00000 | 0.0    | 0.000000  | 0.000000 |
    | Kiss Me, Guido (1997)                       | 0.000000 | 0.0      | 0.0     | 0.0       | 0.0     | 0.000000 | 0.00000 | 0.0    | 0.000000  | 0.000000 |
    | Transformers: Dark of the Moon (2011)       | 0.000000 | 0.0      | 0.0     | 0.0       | 0.0     | 0.316238 | 0.50147 | 0.0    | 0.000000  | 0.605623 |
    | Love (2015)                                | 0.000000 | 0.0      | 0.0     | 0.0       | 0.0     | 0.000000 | 0.00000 | 0.0    | 0.000000  | 0.000000 |
    | Laura (1944)                               | 0.000000 | 0.0      | 0.0     | 0.0       | 0.0     | 0.000000 | 0.00000 | 0.0    | 0.757191  | 0.000000 |
    | Sympathy for the Underdog (1971)            | 0.000000 | 0.0      | 0.0     | 0.0       | 0.0     | 0.598095 | 0.00000 | 0.0    | 0.000000  | 0.000000 |
    | Spellbound (2011)                          | 0.000000 | 0.0      | 0.0     | 0.0       | 0.0     | 0.000000 | 0.00000 | 0.0    | 0.000000  | 0.000000 |
    | Blue Angel, The (Blaue Engel, Der) (1930)   | 0.000000 | 0.0      | 0.0     | 0.0       | 0.0     | 0.000000 | 0.00000 | 0.0    | 0.000000  | 0.000000 |
    | Five Deadly Venoms (1978)                   | 0.000000 | 0.0      | 0.0     | 0.0       | 0.0     | 1.000000 | 0.00000 | 0.0    | 0.000000  | 0.000000 |


    



### Collaborative Filtering

Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Conclusion

## References
[1]   Halim, A., Gohzali, H., Panjaitan, D. M., & Maulana, I. (2017). Sistem Rekomendasi Film menggunakan Bisecting K-Means dan Collaborative Filtering. [Available](https://citisee.amikompurwokerto.ac.id/assets/proceedings/2017/TI08.pdf)

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
