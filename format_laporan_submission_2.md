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

    Periksa fitur yang tidak terlalu berpengaruh terhadap hasil rating. Dari fitur yang ada, terdapat fitur ``timestamp``. fitur ini tidak digunakan, oleh karena itu fitur ini harus dihilangkan. cara menghilangkannya yaitu dengan fungsi ``drop()`` dengan parameter nama kolom yang ingin dihapus. Berikut kodenya:
    
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

    Dari data diatas, terdapat nama genre `'(no genres listed)'`, maka data ini harus dihilangkan, agar data ini tidak dianggap sebagai nama genre.

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



Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
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
