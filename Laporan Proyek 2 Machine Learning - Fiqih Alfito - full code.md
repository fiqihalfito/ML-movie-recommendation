# Laporan Proyek Machine Learning - Fiqih Alfito

## Project Overview

Sistem rekomendasi adalah suatu mekanisme yang dapat memberikan suatu informasi atau rekomendasi sesuai dengan kesukaan user berdasarkan informasi yang diperoleh dari user. Oleh karena itu, diperlukan model rekomendasi yang tepat agar rekomendasi yang diberikan oleh sistem sesuai dengan kesukaan user, serta mempermudah user mengambil keputusan dalam  enentukan item (film) yang akan dipilih.

Sistem rekomendasi film menjadi penting dalam konteks industri hiburan dan digital, di mana jumlah konten film dan TV yang tersedia sangat luas dan terus bertambah. Dalam lingkungan seperti ini, pengguna seringkali menghadapi kesulitan dalam menavigasi melalui pilihan yang melimpah. Sistem rekomendasi film bertujuan untuk membantu pengguna menemukan konten yang paling relevan dan menarik bagi mereka, sehingga memperbaiki pengalaman konsumen dan meningkatkan retensi pengguna. Dengan memberikan rekomendasi yang akurat, sistem ini juga membantu meningkatkan kepuasan pengguna dan memfasilitasi penemuan konten baru yang mungkin tidak akan ditemukan secara konvensional[1].

## Business Understanding

### Problem Statements

Berdasarkan kondisi yang telah diuraikan sebelumnya, penulis akan mengembangkan sebuah sistem prediksi diabetes untuk menjawab permasalahan berikut.

Menjelaskan pernyataan masalah:
- Berdasarkan data mengenai *user*, bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik content-based filtering?
- Dengan data rating yang Anda miliki, bagaimana sistem dapat merekomendasikan movie lain yang mungkin disukai dan belum pernah dikunjungi oleh *user*? 

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Menghasilkan sejumlah rekomendasi restoran yang dipersonalisasi untuk pengguna dengan teknik content-based filtering.
- Menghasilkan sejumlah rekomendasi movie yang sesuai dengan preferensi pengguna dan belum pernah ditonton sebelumnya dengan teknik collaborative filtering.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.


### Solution statements

Untuk mencapai goals yang diinginkan, penulis akan membuat sistem rekomendasi dengan alur sebagai berikut.

1. Data Understanding

    Data Understanding adalah tahap awal proyek untuk memahami data yang dimiliki. Dalam kasus ini, penulis memiliki 2 file terpisah, ratings.csv dan movies.csv.

2. Univariate Exploratory Data Analysis

    Pada tahap ini, penulis melakukan analisis dan eksplorasi setiap variabel pada data. setiap fitur akan dianalisis apakah fitur tersebut akan berpengaruh atau tidak pada model.

3. Data Preprocessing

    Ini merupakan tahap persiapan data sebelum data digunakan untuk proses selanjutnya. Pada tahap ini, penulis akan melakukan penggabungan beberapa file sehingga menjadi satu kesatuan file yang utuh dan siap digunakan dalam tahap pemodelan.

4. Data Preparation

    Pada tahap ini, penulis mempersiapkan data dan melakukan beberapa teknik seperti mengatasi missing value, drop fitur yang tidak digunakan, dan hapus movie yang tidak memiliki genre. Selanjutnya data telah siap untuk digunakan dalam pengembangan model. 

5. Model Development dengan Content Based Filtering

    Pada tahap inilah penulis mengembangkan sistem rekomendasi dengan teknik content based filtering. Teknik content based filtering akan merekomendasikan movie yang mirip dengan movie yang dinilai/rating *user* di masa lalu. Pada tahap ini, Penulis akan menemukan representasi fitur penting dari setiap genre movie dengan tfidf vectorizer dan menghitung tingkat kesamaan dengan cosine similarity. Setelah itu, penulis akan membuat sejumlah rekomendasi movie untuk user berdasarkan kesamaan yang telah dihitung sebelumnya.

6. Model Development dengan Collaborative Filtering

    Pada tahap ini, sistem merekomendasikan sejumlah movie berdasarkan rating yang telah diberikan sebelumnya. Dari data rating *user*, penulis akan mengidentifikasi movie-movie yang mirip dan belum pernah ditonton oleh *user* untuk direkomendasikan.

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

    Tabel 1. total rating setiap movie.

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

    maka fitur ``timestamp`` akan hilang.

    Tabel 2. tabel rating setelah fitur `timestamp` dihapus.

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

    Tabel 3. tabel movie yang berisi tidak ada genre.
    
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

### **Content Based Filtering**

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

    Tabel 4. Tabel matrix tf-idf untuk beberapa movie dan genre

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


2. **Cosine Similarity**

    Cosine similarity adalah metode yang umum digunakan dalam sistem rekomendasi untuk mengukur tingkat kesamaan antara dua vektor. Ini membantu dalam menentukan sejauh mana dua item atau entitas serupa atau memiliki preferensi yang serupa.

    Dalam konteks sistem rekomendasi, item atau entitas yang akan direkomendasikan sering kali diwakili sebagai vektor dalam ruang fitur. Setiap dimensi dalam vektor ini mewakili atribut atau fitur yang relevan untuk item tersebut. Misalnya, dalam sistem rekomendasi film, setiap film dapat direpresentasikan sebagai vektor dengan dimensi yang mewakili atribut-atribut seperti genre, aktor, sutradara, dll.

    Cosine similarity menggunakan konsep dari ruang vektor untuk mengukur sudut antara dua vektor. Semakin dekat sudut antara dua vektor dengan 0 derajat (kemungkinan terbesar), semakin mirip atau serupa vektor-vektor tersebut. Metode ini mengabaikan magnitudo vektor dan hanya memperhatikan arahnya.

    Dalam sistem rekomendasi, cosine similarity dapat digunakan untuk menghitung kesamaan antara dua item atau entitas berdasarkan atribut-atribut yang relevan. Dengan menghitung cosine similarity antara vektor representasi item yang ada dalam dataset dengan vektor representasi item yang sedang direkomendasikan, kita dapat mengidentifikasi item yang paling mirip atau serupa dengan item yang sedang dicari. Item dengan nilai cosine similarity yang lebih tinggi kemungkinan besar akan menjadi rekomendasi yang lebih relevan bagi pengguna.

    Hasil dari cosine similarity akan berada dalam rentang -1 hingga 1. Nilai 1 menunjukkan kesamaan sempurna antara dua vektor, sedangkan nilai -1 menunjukkan perbedaan sempurna. Nilai 0 menunjukkan bahwa dua vektor saling tegak lurus atau tidak ada kesamaan sama sekali.

    Dalam proyek ini, masukkan *td-idf matrix* yang didapat pada tahap TF-IDF Vectorizer sebelumnya sebagai parameter dari fungsi ``cosine_similarity()``.

    ```py
    # Menghitung cosine similarity pada matrix tf-idf
    cosine_sim = cosine_similarity(tfidf_matrix)
    cosine_sim
    ```

    Untuk melihat hasil perhitungan *cosine similarity*, buat DataFrame dengan variabel `cosine_sim` sebagai data, `title` sebagai kolom dan baris. Maka akan terlihat derajat kesamaan antar *movie*.

    Tabel 5. Perhitungan *cosine similarity* antar *title movie.*

    | title                                              | Japanese Story (2003) | I Am Trying to Break Your Heart (2002) | Mouchette (1967) | Deceiver (1997) | Little Princess, A (1995) | Picture of Dorian Gray, The (1945) | Failure to Launch (2006) | 4 Little Girls (1997) | Stand and Deliver (1988) | Lonely Are the Brave (1962) |
    |----------------------------------------------------|-----------------------|----------------------------------------|------------------|-----------------|---------------------------|-----------------------------------|--------------------------|-----------------------|---------------------------|-----------------------------|
    | The House (2017)                                   | 0.0                   | 0.0                                    | 0.0              | 0.0             | 0.0                       | 0.0                               | 0.570705                 | 0.0                   | 0.734682                  | 0.0                         |
    | RocknRolla (2008)                                  | 0.0                   | 0.0                                    | 0.0              | 0.52661         | 0.0                       | 0.0                               | 0.0                      | 0.0                   | 0.0                       | 0.0                         |
    | City of Women, The (Città delle donne, La) (1980)  | 0.678412              | 0.0                                    | 0.678412         | 0.274935        | 0.298034                  | 0.237259                          | 0.419287                 | 0.0                   | 1.0                       | 0.227514                    |
    | John Mulaney: New In Town (2012)                   | 0.0                   | 0.0                                    | 0.0              | 0.0             | 0.0                       | 0.0                               | 0.570705                 | 0.0                   | 0.734682                  | 0.0                         |
    | Time of the Gypsies (Dom za vesanje) (1989)        | 0.334307              | 0.0                                    | 0.334307         | 0.534875        | 0.146865                  | 0.564351                          | 0.206615                 | 0.0                   | 0.492778                  | 0.112114                    |
    | Marvel One-Shot: Agent Carter (2013)               | 0.0                   | 0.0                                    | 0.0              | 0.0             | 0.0                       | 0.382864                          | 0.0                      | 0.0                   | 0.0                       | 0.0                         |
    | Open Season (2006)                                 | 0.0                   | 0.0                                    | 0.0              | 0.0             | 0.404338                  | 0.0                               | 0.136037                 | 0.0                   | 0.175124                  | 0.0                         |
    | 12 Chairs (1976)                                   | 0.0                   | 0.0                                    | 0.0              | 0.0             | 0.0                       | 0.0                               | 0.308159                 | 0.0                   | 0.3967                    | 0.0                         |
    | Rebound (2005)                                     | 0.0                   | 0.0                                    | 0.0              | 0.0             | 0.0                       | 0.0                               | 0.570705                 | 0.0                   | 0.734682                  | 0.0                         |
    | Married to the Mob (1988)                          | 0.0                   | 0.0                                    | 0.0              | 0.0             | 0.0                       | 0.0                               | 0.570705                 | 0.0                   | 0.734682                  | 0.0                         |


3. **Mendapatkan Rekomendasi**

    Untuk buat rekomendasi, buatlah fungsi untuk memberikan rekomendasi dengan parameter sebagai berikut:

    - title : judul movie (index kemiripan dataframe)
    - similarity_data : DataFrame mengenai similarity yang telah kita definisikan sebelumnya.
    - items : data movie yang akan digunakan untuk mendefinisikan kemiripan.
    - k : banyak rekomendasi yang ingin diberikan.

    sebelum mulai menulis kodenya, ingatlah kembali definisi sistem rekomendasi yang menyatakan bahwa keluaran sistem ini adalah berupa top-N recommendation. Oleh karena itu, kita akan memberikan sejumlah rekomendasi movie pada pengguna yang diatur dalam parameter k.

    ```py
    def movie_recommendations(title, similarity_data=cosine_sim_df, items=data, k=5):
    
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,title].to_numpy().argpartition(
        range(-1, -k, -1))

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    # Drop title agar nama title yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(title, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)
    ```

    Kemudian ambil satu sampel movie untuk diberikan rekomendasi pada movie tersebut.

    ```py
    data[data.title.eq('Unleashed (Danny the Dog) (2005)')]
    ```

    Tabel 6. sampel movie sebagai rujukan untuk rekomendasi.

    | movieId | title                             | genres                      |
    |---------|-----------------------------------|-----------------------------|
    | 33437   | Unleashed (Danny the Dog) (2005)  | Action/Crime/Drama/Thriller |

    Selanjutnya jalankan fungsi `movie_recommendation()` dengan parameter judul movie seperti sampel sebelumnya.

    ```py
    movie_recommendations('Unleashed (Danny the Dog) (2005)')
    ```
    
    maka akan tampil tabel berisi 5 rekomendasi teratas dari judul movie yang kita berikan. 5 rekomendasi judul movie adalah nilai *default* parameter `k`, nilai `k` dapat diubah.

    Tabel 7. Hasil rekomendasi Content-Based Filtering.
    
    | title                              | movieId | genres                            |
    |------------------------------------|---------|-----------------------------------|
    | Collateral (2004)                  | 8798    | Action/Crime/Drama/Thriller       |
    | To Live and Die in L.A. (1985)     | 7040    | Action/Crime/Drama/Thriller       |
    | The Fate of the Furious (2017)     | 170875  | Action/Crime/Drama/Thriller       |
    | Pusher II: With Blood on My Hands (2004) | 34811   | Action/Crime/Drama/Thriller       |
    | Bullitt (1968)                     | 7076    | Action/Crime/Drama/Thriller       |
  
    Dari tabel di atas bahwa movie "Unleashed (Danny the Dog) (2005)" dengan kategori Action|Crime|Drama|Thriller menghasilkan rekomendasi movie dengan kategori yang sama.


### **Collaborative Filtering**

Collaborative Filtering (CF) adalah salah satu metode yang digunakan dalam sistem rekomendasi untuk memberikan rekomendasi kepada pengguna berdasarkan preferensi atau perilaku pengguna lainnya. Pendekatan ini didasarkan pada asumsi bahwa pengguna yang memiliki preferensi yang sama di masa lalu cenderung memiliki preferensi yang sama di masa depan.

Ada dua jenis utama dari Collaborative Filtering:

- **Collaborative Filtering berdasarkan Item (Item-Based Collaborative Filtering)**: Metode ini menganalisis hubungan antara item yang berbeda dalam sistem rekomendasi. Jika dua item sering kali diberikan peringkat yang sama oleh pengguna, maka item tersebut dianggap memiliki hubungan dan dianggap saling terkait. Ketika seorang pengguna memberikan peringkat pada satu item, sistem rekomendasi dapat merekomendasikan item lain yang memiliki hubungan yang kuat dengan item yang diberi peringkat tersebut.

- **Collaborative Filtering berdasarkan Pengguna (User-Based Collaborative Filtering)**: Metode ini menganalisis kesamaan preferensi antara pengguna. Jika dua pengguna sering memiliki preferensi yang serupa dalam memberikan peringkat pada item, maka sistem rekomendasi dapat merekomendasikan item yang telah diberi peringkat tinggi oleh satu pengguna kepada pengguna lain yang memiliki kesamaan preferensi.

Kelebihan Collaborative Filtering:

- Mampu memberikan rekomendasi yang personal: Collaborative Filtering dapat memberikan rekomendasi yang sesuai dengan preferensi pengguna secara individu. Ini karena rekomendasi didasarkan pada perilaku atau preferensi pengguna lain yang memiliki kesamaan dengan pengguna yang bersangkutan.

- Tidak memerlukan informasi detil tentang item: CF tidak memerlukan informasi rinci tentang setiap item dalam sistem rekomendasi. Itu hanya membutuhkan data tentang preferensi atau peringkat pengguna terhadap item yang tersedia.

Kekurangan Collaborative Filtering:

- Masalah Cold Start: Metode ini menghadapi masalah saat pengguna baru bergabung dengan sistem rekomendasi atau ketika item baru diperkenalkan. Karena tidak ada data historis tentang pengguna baru atau item baru, Collaborative Filtering kesulitan memberikan rekomendasi yang akurat.

- Scalability: Jika sistem rekomendasi memiliki jumlah pengguna dan item yang sangat besar, perhitungan kesamaan antara pengguna atau item dapat menjadi komputasi yang rumit dan memakan waktu. Skalabilitas bisa menjadi masalah dalam implementasi sistem rekomendasi berbasis Collaborative Filtering.

- Masalah Keberagaman: Collaborative Filtering cenderung memperkuat preferensi yang ada dan mengabaikan rekomendasi yang berbeda atau inovatif. Hal ini dapat menyebabkan pengguna terperangkap dalam "filter bubble" di mana mereka hanya menerima rekomendasi yang serupa dengan apa yang mereka sukai sebelumnya.

Berikut adalah tahapan dari Collaborative Filtering:

1. Data Understanding

    Mengambil data `ratings` sebagai data untuk dijadikan rekomendasi.

    ```py
    df = ratings.copy()
    ```

2. Data Preparation

    Pada tahap ini, kita perlu melakukan persiapan data untuk menyandikan (encode) fitur `userId` dan `movieId` ke dalam indeks integer.

    ```py
    # Mengubah userId menjadi list tanpa nilai yang sama
    user_ids = df['userId'].unique().tolist()
    print('list userId: ', user_ids)

    # Melakukan encoding userID
    user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
    print('encoded userId : ', user_to_user_encoded)

    # Melakukan proses encoding angka ke ke userId
    user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
    print('encoded angka ke userId: ', user_encoded_to_user) 
    ```

    Kemudian petakan `userId` dan `movieId` ke dataframe `ratings`.

    Tabel 8. pemetaan *encode* *user* dan *movie* ke *dataframe* `ratings`

    | userId | movieId | rating | user | movie |
    |--------|---------|--------|------|-------|
    |   1    |    1    |  4.0   |   0  |   0   |
    |   1    |    3    |  4.0   |   0  |   1   |
    |   1    |    6    |  4.0   |   0  |   2   |
    |   1    |   47    |  5.0   |   0  |   3   |
    |   1    |   50    |  5.0   |   0  |   4   |
    |   ...  | 

    Terakhir, cek beberapa hal dalam data seperti jumlah user dan jumlah movie.

    ```py
    # Mendapatkan jumlah user
    num_users = len(user_to_user_encoded)
    print(num_users)

    # Mendapatkan jumlah movie
    num_movies = len(movie_to_movie_encoded)
    print(num_movies)

    # Nilai minimum rating
    min_rating = min(df['rating'])

    # Nilai maksimal rating
    max_rating = max(df['rating'])

    print('Number of User: {}, Number of Movie: {}, Min Rating: {}, Max Rating: {}'.format(
        num_users, num_movies, min_rating, max_rating
    ))
    ```

    ```
    610
    9724
    Number of User: 610, Number of Movie: 9724, Min Rating: 0.5, Max Rating: 5.0
    ```

3. Membagi Data untuk Training dan Validasi

    Sebelum membagi data training dan validasi, acak datanya terlebih dahulu agar distribusinya random.

    ```py
    # Mengacak dataset
    df = df.sample(frac=1, random_state=42)
    ```

    Karena data berjumlah banyak, maka kita bagi data dengan rasio 90:10. Namun sebelumnya, kita perlu memetakan data `user` dan `movie` menjadi satu value terlebih dahulu. Lalu, buatlah rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training.

    ```py
    # Membuat variabel x untuk mencocokkan data user dan post menjadi satu value
    x = df[['user', 'movie']].values

    # Membuat variabel y untuk membuat rating menjadi skala 0 sampai 1
    y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

    # Membagi menjadi 90% data train dan 10% data validasi
    train_indices = int(0.9 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:]
    )
    ```

    Selanjutnya data telah siap untuk dimasukkan ke model.

4. Proses Training

    Pada tahap ini, model menghitung skor kecocokan antara pengguna dan movie dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan movie. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan movie. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan movie. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.

    Di sini, kita membuat class RecommenderNet dengan keras Model class. Kode class RecommenderNet ini terinspirasi dari tutorial dalam situs Keras dengan beberapa adaptasi sesuai kasus yang sedang kita selesaikan. Terapkan kode berikut.

    ```py
    class RecommenderNet(tf.keras.Model):

    # Insialisasi fungsi
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = keras.layers.Embedding( # layer embedding user
            num_users,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer = keras.regularizers.l2(1e-6)
        )
        self.user_bias = keras.layers.Embedding(num_users, 1) # layer embedding user bias
        self.movie_embedding = keras.layers.Embedding( # layer embeddings movie
            num_movies,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer = keras.regularizers.l2(1e-6)
        )
        self.movie_bias = keras.layers.Embedding(num_movies, 1) # layer embedding movie bias

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
        user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
        movie_vector = self.movie_embedding(inputs[:, 1]) # memanggil layer embedding 3
        movie_bias = self.movie_bias(inputs[:, 1]) # memanggil layer embedding 4

        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)

        x = dot_user_movie + user_bias + movie_bias

        return tf.nn.sigmoid(x) # activation sigmoid
    ```

    Selanjutnya, lakukan proses compile terhadap model.

    ```py
    model = RecommenderNet(num_users, num_movies, 50) # inisialisasi model

    # model compile
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    ```

    Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. 

    Langkah berikutnya, mulailah proses training.

    ```py
    # Memulai training

    history = model.fit(
        x = x_train,
        y = y_train,
        batch_size = 64,
        epochs = 5,
        validation_data = (x_val, y_val)
    )
    ``` 

    Setelah melakukan fit model, maka model siap digunakan untuk menghasilkan rekomendasi.

5. Mendapatkan Rekomendasi Movie

    Untuk mendapatkan rekomendasi movie, pertama kita ambil sampel user secara acak dan definisikan variabel movie_not_watched yang merupakan daftar movie yang belum pernah dikunjungi oleh pengguna. Anda mungkin bertanya-tanya, mengapa kita perlu menentukan daftar movie_not_watched? Hal ini karena daftar movie_not_watched inilah yang akan menjadi movie yang kita rekomendasikan. 

    Sebelumnya, pengguna telah memberi rating pada beberapa movie yang telah mereka kunjungi. Kita menggunakan rating ini untuk membuat rekomendasi movie yang mungkin cocok untuk pengguna. Nah, movie yang akan direkomendasikan tentulah movie yang belum pernah dikunjungi oleh pengguna. Oleh karena itu, kita perlu membuat variabel movie_not_watched sebagai daftar movie untuk direkomendasikan pada pengguna.

    Variabel movie_not_watched diperoleh dengan menggunakan operator bitwise (~) pada variabel movie_watched_by_user.

    Terapkan kode berikut.

    ```py
    movie_df = fix_movies.copy()
    df = pd.read_csv('ratings.csv')

    # Mengambil sample user
    user_id = df.userId.sample(1).iloc[0]
    movie_watched_by_user = df[df.userId == user_id]

    # Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html
    movie_not_watched = movie_df[~movie_df['movieId'].isin(movie_watched_by_user.movieId.values)]['movieId']
    movie_not_watched = list(
        set(movie_not_watched)
        .intersection(set(movie_to_movie_encoded.keys()))
    )

    movie_not_watched = [[movie_to_movie_encoded.get(x)] for x in movie_not_watched]
    movie_not_watched
    user_encoder = user_to_user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movie_not_watched), movie_not_watched)
    )
    ```

    Selanjutnya, untuk memperoleh rekomendasi restoran, gunakan fungsi model.predict() dari library Keras dengan menerapkan kode berikut.

    ```py
    ratings = model.predict(user_movie_array).flatten()

    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
        movie_encoded_to_movie.get(movie_not_watched[x][0]) for x in top_ratings_indices
    ]

    print('Showing recommendations for users: {}'.format(user_id))
    print('===' * 9)
    print('movie with high ratings from user')
    print('----' * 8)

    top_movie_user = (
        movie_watched_by_user.sort_values(
            by = 'rating',
            ascending=False
        )
        .head(5)
        .movieId.values
    )

    movie_df_rows = movie_df[movie_df['movieId'].isin(top_movie_user)]
    for idx, row in enumerate(movie_df_rows.itertuples(index=False), start=1):
        print("{}. Title:".format(idx), row[1])
        print("   Genres:", row[2])
        print()

    print('----' * 8)
    print('Top 10 movie recommendation')
    print('----' * 8)

    recommended_movie = movie_df[movie_df['movieId'].isin(recommended_movie_ids)]
    for idx, row in enumerate(recommended_movie.itertuples(index=False), start=1):
        print("{}. Title:".format(idx), row[1])
        print("   Genres:", row[2])
        print()
    ```

    Berikut rekomendasi yang dihasilkan dari model.

    ```
    Showing recommendations for users: 425
    ===========================
    movie with high ratings from user
    --------------------------------
    1. Title: Léon: The Professional (a.k.a. The Professional) (Léon) (1994)
    Genres: Action|Crime|Drama|Thriller

    2. Title: Shawshank Redemption, The (1994)
    Genres: Crime|Drama

    3. Title: Forrest Gump (1994)
    Genres: Comedy|Drama|Romance|War

    4. Title: Trainspotting (1996)
    Genres: Comedy|Crime|Drama

    5. Title: Sleepers (1996)
    Genres: Thriller

    --------------------------------
    Top 10 movie recommendation
    --------------------------------
    1. Title: Rear Window (1954)
    Genres: Mystery|Thriller

    2. Title: Princess Bride, The (1987)
    Genres: Action|Adventure|Comedy|Fantasy|Romance

    3. Title: Goodfellas (1990)
    Genres: Crime|Drama

    4. Title: Amadeus (1984)
    Genres: Drama

    5. Title: Raging Bull (1980)
    Genres: Drama

    6. Title: Boot, Das (Boat, The) (1981)
    Genres: Action|Drama|War

    7. Title: Glory (1989)
    Genres: Drama|War

    8. Title: Graduate, The (1967)
    Genres: Comedy|Drama|Romance

    9. Title: Chinatown (1974)
    Genres: Crime|Film-Noir|Mystery|Thriller

    10. Title: Cool Hand Luke (1967)
    Genres: Drama
   ```

    Hasil di atas adalah rekomendasi untuk user dengan id 425. Dari output tersebut, kita dapat membandingkan antara Movie with high ratings from user dan Top 10 movie recommendation untuk user. 

    Perhatikanlah, beberapa movie rekomendasi menyediakan genre yang sesuai dengan rating user. Kita memperoleh movie dengan genre dominan Drama, sesuai dengan rating user yang lebih dominan movie bergenre Drama. Begitu pula dengan genre lainnya. Prediksi yang cukup sesuai.

## Evaluation

Dalam proyek ini, kedua solusi rekomendasi mempunyai masing-masing cara evaluasi. Berikut evaluasi dari solusi-solusi rekomendasi. 

### **Evaluasi Content Based Filtering**

Untuk mengevaluasi hasil rekomendasi Content Based Filtering, penulis menggunakan Precision. Metrik ini mengukur sejauh mana rekomendasi yang diberikan relevan dengan preferensi pengguna. Presisi dihitung dengan membagi jumlah rekomendasi yang relevan dengan jumlah total rekomendasi yang diberikan.

```
Presisi = ((Jumlah Rekomendasi yang Relevan) / (Jumlah Total Rekomendasi)) * 100%
```

Pada proyek ini, sampel movie yang digunakan sebagai berikut:

Tabel 9. sampel movie sebagai rujukan untuk rekomendasi.

| movieId | title                             | genres                      |
|---------|-----------------------------------|-----------------------------|
| 33437   | Unleashed (Danny the Dog) (2005)  | Action/Crime/Drama/Thriller |

Setelah dijalankan fungsi `movie_recommendation()`, maka didapat rekomendasi movie berikut:

Tabel 10. Hasil rekomendasi Content-Based Filtering.

| title                              | movieId | genres                            |
|------------------------------------|---------|-----------------------------------|
| Collateral (2004)                  | 8798    | Action/Crime/Drama/Thriller       |
| To Live and Die in L.A. (1985)     | 7040    | Action/Crime/Drama/Thriller       |
| The Fate of the Furious (2017)     | 170875  | Action/Crime/Drama/Thriller       |
| Pusher II: With Blood on My Hands (2004) | 34811   | Action/Crime/Drama/Thriller       |
| Bullitt (1968)                     | 7076    | Action/Crime/Drama/Thriller       |

Dari tabel diatas, diketahui bahwa :

```
Jumlah rekomendasi yang relevan = 5;
Jumlah total rekomendasi = 5;
```

Selanjutnya, untuk mengukur evaluasi dari rekomendasi ini, penulis menggunakan rumus presisi.

```
Presisi = ((Jumlah Rekomendasi yang Relevan) / (Jumlah Total Rekomendasi)) * 100%

Presisi = (5 / 5) * 100%

Presisi = 100%
```

Maka dapat diketahui bahwa nilai evaluasi Content Based Filtering menggunakan metrik Precision adalah 100%.

### **Evaluasi Collaborative Filtering**

Untuk mengevaluasi model yang telah dilatih, proyek ini menggunakan metrik `root mean squared error (RMSE)`. Root Mean Squared Error (RMSE) adalah metrik evaluasi yang digunakan untuk mengukur tingkat kesalahan atau deviasi dari prediksi model dalam perbandingan dengan nilai sebenarnya pada masalah regresi pada machine learning. RMSE merupakan akar kuadrat dari Mean Squared Error (MSE).

- **Metrik Root Mean Squared Error (RMSE)** 

    Root Mean Squared Error (RMSE) adalah metrik evaluasi yang umum digunakan dalam sistem rekomendasi untuk mengukur sejauh mana prediksi rekomendasi yang dibuat oleh sistem mendekati nilai yang sebenarnya atau preferensi pengguna. RMSE menghitung perbedaan antara nilai prediksi dan nilai sebenarnya dari set data pengujian.

    Rumus RMSE adalah sebagai berikut:

    ```
    RMSE = sqrt((1/n) * Σ(yi - ŷi)^2)
    ```

    Di sini:
    - n adalah jumlah data pengujian.
    - yi adalah nilai sebenarnya.
    - ŷi adalah nilai prediksi.

    Cara kerja RMSE dalam sistem rekomendasi adalah sebagai berikut:

    1. Membangun model: Sistem rekomendasi memanfaatkan berbagai algoritma seperti Collaborative Filtering, Content-Based Filtering, atau Hybrid Filtering untuk membangun model rekomendasi. Model ini didasarkan pada data pelatihan yang mencakup preferensi pengguna sebelumnya.

    2. Menghitung prediksi: Setelah model dibangun, sistem rekomendasi menggunakan model tersebut untuk membuat prediksi tentang preferensi pengguna untuk item-item yang belum dilihat atau dinilai. Prediksi ini didasarkan pada fitur-fitur atau atribut-atribut item serta informasi pengguna yang relevan.

    3. Membandingkan dengan nilai sebenarnya: Setelah mendapatkan prediksi, sistem rekomendasi membandingkannya dengan nilai sebenarnya atau preferensi yang diketahui dari data pengujian. Selisih antara prediksi dan nilai sebenarnya untuk setiap item dihitung.

    4. Menghitung RMSE: Selisih antara prediksi dan nilai sebenarnya untuk setiap item dijumlahkan dan dinormalisasi dengan membaginya dengan jumlah data pengujian (n). Akar kuadrat dari hasil normalisasi ini memberikan RMSE, yang merupakan pengukuran kesalahan rata-rata antara prediksi dan nilai sebenarnya.

    RMSE memberikan informasi tentang seberapa dekat prediksi sistem rekomendasi dengan preferensi pengguna yang sebenarnya. Semakin kecil nilai RMSE, semakin akurat sistem rekomendasi dalam melakukan prediksi dan semakin baik kualitas rekomendasinya. Dalam pengembangan sistem rekomendasi, tujuan utama adalah untuk meminimalkan RMSE agar prediksi semakin mendekati nilai sebenarnya.

- **Visualisasi Metrik**

    Untuk melihat visualisasi proses training, mari kita plot metrik evaluasi dengan matplotlib. Terapkan kode berikut.

    ```py
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('model_metrics')
    plt.ylabel('root_mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    ```

    maka akan tampil gambar berikut:

    ![visual metrik](https://github.com/fiqihalfito/ML-movie-recommendation/assets/112739333/5a471719-b832-4675-bcfc-7d34a2acb148)

    Gambar 1. Visualisasi Metrik Collaborative Filtering


    Pada hasil fit model, RMSE dihitung untuk data pelatihan (train) dan data validasi (validation) setiap epoch. Berikut adalah ringkasan hasil RMSE pada setiap epoch:

    - Pada Epoch 1, RMSE pada data pelatihan adalah 0.2322 dan pada data validasi adalah 0.2148.
    - Pada Epoch 2, RMSE pada data pelatihan turun menjadi 0.2081 dan pada data validasi turun menjadi 0.2118.
    - Pada Epoch 3, RMSE pada data pelatihan menjadi 0.2029 dan pada data validasi menjadi 0.2081.
    - Pada Epoch 4, RMSE pada data pelatihan menjadi 0.2016 dan pada data validasi menjadi 0.2071.
    - Pada Epoch 5, RMSE pada data pelatihan menjadi 0.2010 dan pada data validasi menjadi 0.2055.

    Dari hasil RMSE tersebut, dapat dilihat bahwa baik pada data pelatihan maupun data validasi, RMSE mengalami penurunan setiap epoch. Hal ini menunjukkan bahwa model semakin mempelajari pola dalam data dan semakin baik dalam memprediksi nilai target.

    Selain itu, RMSE pada data validasi juga memberikan gambaran tentang performa model pada data yang tidak digunakan dalam proses pelatihan. Jika RMSE pada data validasi tetap rendah dan mendekati RMSE pada data pelatihan, maka model memiliki kemampuan umum dalam memprediksi rating tanpa overfitting pada data pelatihan.


## Conclusion

Pada solusi Content Based Filtering, solusi ini menghasilkan rekomendasi yang sangat sesuai dengan sampel movie yang diberikan berdasarkan fitur genre movie.

Pada solusi Collaborative Filtering, solusi ini menghasilkan rekomendasi yang relevan terhadap rating user dan genre. Rekomendasi movie yang diberikan memiliki genre yang sama paling tidak satu genre yang sama.

Berdasarkan hasil visualisasi metrik, proses training model cukup smooth dan model konvergen pada epochs sekitar 4. Dari proses ini, kita memperoleh nilai error (RSME) akhir sebesar sekitar 0.20 dan error pada data validasi sebesar 0.20. Nilai tersebut cukup bagus untuk sistem rekomendasi. 

Dalam konteks RMSE, epoch terakhir mengacu pada hasil pelatihan model pada epoch ke-5. Pada epoch terakhir ini, RMSE dihitung untuk data pelatihan dan data validasi.

- RMSE pada data pelatihan pada epoch terakhir adalah 0.2010. Ini berarti, pada saat itu, model rata-rata memprediksi rating dengan kesalahan sekitar 0.2010. Semakin rendah nilai RMSE, semakin akurat model dalam memprediksi rating.

- RMSE pada data validasi pada epoch terakhir adalah 0.2055. Ini menunjukkan kesalahan rata-rata model dalam memprediksi rating pada data validasi sekitar 0.2055. 

Dalam kasus ini, RMSE pada data validasi hampir sebanding dengan RMSE pada data pelatihan, yang menunjukkan bahwa model tidak mengalami overfitting (yaitu, model tidak terlalu mengoptimalkan performa pada data pelatihan dan kehilangan kemampuan umum dalam memprediksi rating pada data baru).

Dalam kesimpulannya, pada epoch terakhir, model sistem rekomendasi memiliki nilai RMSE sebesar 0.2010 pada data pelatihan dan 0.2055 pada data validasi. Nilai-nilai ini menggambarkan kesalahan rata-rata dalam memprediksi rating dan memberikan indikasi tentang seberapa akurat model dalam memberikan rekomendasi kepada pengguna. Semakin rendah nilai RMSE, semakin akurat model dalam memprediksi nilai target dan semakin dekat prediksi model dengan nilai sebenarnya.

Saran untuk kedepannya, dataset movie bisa ditambahkan fitur-fitur lainnya agar bisa diimplementasikan ke real project.

## References
[1]   Halim, A., Gohzali, H., Panjaitan, D. M., & Maulana, I. (2017). Sistem Rekomendasi Film menggunakan Bisecting K-Means dan Collaborative Filtering. [Available](https://citisee.amikompurwokerto.ac.id/assets/proceedings/2017/TI08.pdf)
