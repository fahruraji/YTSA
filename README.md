
  <div align="center">
  <h1 align="center">Youtube Comment Sentiment Analysis</h1>
  <h3>FAHRURAJI - NPM. 19630940</h3>
  <h5>PROGRAM STUDI S1 TEKNIK INFORMATIKA FAKULTAS TEKNOLOGI INFORMASI<br/>
  UNIVERSITAS ISLAM KALIMANTAN MUHAMMAD ARSYAD AL BANJARI BANJARMASIN</h5>

  <p align="center"><img src="https://img.shields.io/badge/-Flask-004E89?logo=Flask&style=plastic" alt='Flask\' />
<img src="https://via.placeholder.com/1/0000/00000000" alt="spacer" /><img src="https://img.shields.io/badge/-googleapiclient-004E89?logo=googleapiclient&style=plastic" alt='googleapiclient\' />
<img src="https://via.placeholder.com/1/0000/00000000" alt="spacer" /><img src="https://img.shields.io/badge/-SQLAlchemy-004E89?logo=SQLAlchemy&style=plastic" alt='SQLAlchemy\' />
<img src="https://via.placeholder.com/1/0000/00000000" alt="spacer" /><img src="https://img.shields.io/badge/-PyMySQL-004E89?logo=PyMySQL&style=plastic" alt='PyMySQL\' />
<img src="https://via.placeholder.com/1/0000/00000000" alt="spacer" /><img src="https://img.shields.io/badge/-NLTK-004E89?logo=NLTK&style=plastic" alt='NLTK\' />
<img src="https://via.placeholder.com/1/0000/00000000" alt="spacer" /><img src="https://img.shields.io/badge/-sastrawi-004E89?logo=sastrawi&style=plastic" alt='sastrawi\' />
<img src="https://via.placeholder.com/1/0000/00000000" alt="spacer" /><img src="https://img.shields.io/badge/-scikit%20learn-004E89?logo=scikit%20learn&style=plastic" alt='scikit-learn\' />
<img src="https://via.placeholder.com/1/0000/00000000" alt="spacer" /><img src="https://img.shields.io/badge/-PyTorch-004E89?logo=PyTorch&style=plastic" alt='PyTorch\' />
<img src="https://via.placeholder.com/1/0000/00000000" alt="spacer" /><img src="https://img.shields.io/badge/-transformers-004E89?logo=transformers&style=plastic" alt='transformers\"' />
<img src="https://via.placeholder.com/1/0000/00000000" alt="spacer" />
  </p>
  </div>
  
  ---
  
  Aplikasi Analisis Sentimen Komentar Youtube ini dibuat untuk membantu mengekstrak opini yang berasal dari komentar para pengguna (viewer) terhadap suatu konten video yang diupload di Youtube dan mengidentifikasi kecenderungan sentimen publik (beserta intensi dari komentar) menanggapi materi atau isu yang dibahas dalam video tersebut. 
  
  Dibangun menggunakan bahasa pemrograman Python dengan microframework Flask, memanfaatkan teknik Natural Language Processing (NLP) untuk melakukan analisis sentimen dan deteksi niat (intensi) pada teks berbahasa Indonesia. Pendekatan utama yang digunakan untuk melakukan analisis adalah pendekatan Transfer Learning menggunakan <strong>Pretrained Language Models IndoBERT</strong>. Sebagai pembanding ditampilkan pula hasil prediksi pendekatan pembelajaran mesin menggunakan <strong>Support Vector Machine (SVM)</strong> dengan pembobotan TF-IDF.

---

## 📁 Struktur Project

```sh
├── app
│   ├── controller
│   │   ├── AuthController.py
│   │   ├── HomeController.py
│   │   ├── MainController.py
│   │   ├── UserController.py
│   │   ├── WordsController.py
│   ├── helpers
│   │   ├── classifying.py
│   │   ├── commons.py
│   │   ├── preprocessing.py
│   │   ├── scraping.py
│   │   ├── visualizing.py
│   ├── ml
│   │   ├── indobert
│   │   │   ├── intent
│   │   │   └── sentiment
│   │   └── svm
│   │       ├── intent
│   │       └── sentiment
│   ├── model
│   │   ├── analysis.py
│   │   ├── comments.py
│   │   ├── compound.py
│   │   ├── corpus.py
│   │   ├── informal.py
│   │   ├── negasi.py
│   │   ├── negative.py
│   │   ├── positive.py
│   │   ├── preprocessed.py
│   │   ├── processed.py
│   │   ├── root.py
│   │   ├── stopword.py
│   │   ├── user.py
│   │   ├── youtube.py
│   │   ├── __init__.py
│   ├── res
│   ├── response.py
│   ├── restriction.py
│   ├── routes
│   │   ├── auth.py
│   │   ├── main.py
│   │   ├── user.py
│   │   ├── words.py
│   ├── static
│   │   ├── css
│   │   │   └── style.css
│   │   ├── img
│   │   │   ├── akun_belajar.png
│   │   │   ├── algo.jpeg
│   │   │   ├── apple-touch-icon.png
│   │   │   ├── profile-img
│   │   │   ├── charts
│   │   │   │   ├── indobert
│   │   │   │   │   ├── intent
│   │   │   │   │   └── sentiment
│   │   │   │   └── svm
│   │   │   │       ├── intent
│   │   │   │       └── sentiment
│   │   │   ├── chat.png
│   │   │   ├── email_header.png
│   │   │   ├── favicon.png
│   │   │   ├── hero-bg.jpg
│   │   │   ├── hero-bg.png
│   │   │   ├── how-it-work.png
│   │   │   ├── Live-Chat.png
│   │   │   ├── pie.png
│   │   │   ├── sentiment+analysis.png
│   │   │   ├── wordclouds
│   │   │   │   ├── negatif
│   │   │   │   └── positif
│   │   │   ├── youtube-icon.png
│   │   │   ├── yt.png
│   │   │   └── ytsa.jpg
│   │   ├── js
│   │   │   ├── base.js
│   │   │   └── main.js
│   │   ├── scss
│   │   │   └── Readme.txt
│   ├── templates
│   │   ├── auth
│   │   │   ├── login.html
│   │   │   └── reset_passwd.html
│   │   ├── error_page
│   │   │   ├── forbidden.html
│   │   │   ├── internal_error.html
│   │   │   └── not_found.html
│   │   ├── home
│   │   │   ├── hero.html
│   │   │   ├── how.html
│   │   │   ├── index.html
│   │   │   ├── others.html
│   │   │   ├── what.html
│   │   │   └── why.html
│   │   ├── kosakata
│   │   │   ├── compound.html
│   │   │   ├── corpus.html
│   │   │   ├── formal.html
│   │   │   ├── negation.html
│   │   │   ├── negative.html
│   │   │   ├── positive.html
│   │   │   ├── root.html
│   │   │   └── stopword.html
│   │   ├── layout
│   │   │   ├── base.html
│   │   │   ├── footer.html
│   │   │   ├── header.html
│   │   │   ├── main-nav.html
│   │   │   └── nav.html
│   │   ├── mail
│   │   │   ├── activation.html
│   │   │   ├── invitation.html
│   │   │   └── verification.html
│   │   ├── main
│   │   │   ├── casefolded.html
│   │   │   ├── classified.html
│   │   │   ├── comments.html
│   │   │   ├── dist_freq.html
│   │   │   ├── filtered.html
│   │   │   ├── history.html
│   │   │   ├── index.html
│   │   │   ├── normalized.html
│   │   │   ├── search.html
│   │   │   ├── stemmed.html
│   │   │   ├── summary.html
│   │   │   ├── tokenized.html
│   │   │   ├── train_result.html
│   │   │   ├── vectorized.html
│   │   │   └── wordcloud.html
│   │   ├── print
│   │   │   └── table.html
│   │   ├── tes.html
│   │   └── user
│   │       ├── profile.html
│   │       └── users.html
│   ├── uploadconfig.py
│   ├── utils
│   │   ├── args_helper.py
│   │   ├── conlleval.py
│   │   ├── data_utils.py
│   │   ├── forward_fn.py
│   │   ├── functions.py
│   │   ├── metrics.py
│   └── __init__.py
├── .env
├── config.py
├── init.py
├── requirements.txt
└── server.py
```
Proyek ini mencakup berbagai file dan direktori, masing-masing dengan tujuan spesifiknya sendiri:* `app`: Berisi aplikasi Flask utama dan rute, template, dan aset statis terkait.* `ml`: Berisi model pembelajaran mesin untuk analisis sentimen dan klasifikasi, termasuk model IndoBERT dan model Support Vector Machine (SVM) untuk analisis sentimen dan deteksi niat .* `model`: Berisi kelas-kelas yang mewakili struktur database.* `res`: Berisi berkas csv hasil impor.* `routes`: Berisi definisi rute untuk aplikasi Flask.* `static`: Berisi aset statis seperti gambar, stylesheet CSS dan file JavaScript.* `templates`: Berisi template HTML untuk Aplikasi Flask.* `utils`: Berisi fungsi utilitas untuk prapemrosesan dan analisis data.* `config.py`: Berisi pengaturan konfigurasi untuk aplikasi Flask.* `init.py`: Menginisialisasi aplikasi Flask.* `requirements.txt` : Berisi dependensi untuk proyek.* `server.py`: Merupakan titik masuk utama aplikasi.


<details><summary>\</summary>

| File | Summary |
| ---- | ------- |
| config.py |  The code defines a Config class that sets various configuration variables for an application, including database credentials, JWT secret key, security salt, and mail server settings. |
| init.py |  The code creates a Flask application and its database, initializes the database, creates an admin user, and prints a success message. |
| server.py |  The code defines a Flask application with various routes, blueprints, and functions for handling user authentication, error handling, and data import/export. The primary function of the code is to provide a web-based interface for managing data in a database, including importing and exporting data in CSV format, as well as performing various operations on the data such as text classification and sentiment analysis. |

</details>

---

<details><summary>\app</summary>

| File | Summary |
| ---- | ------- |
| response.py |  The code defines two functions, `success` and `badRequest`, which return a JSON response with the specified values and message, with `success` returning a 200 status code and `badRequest` returning a 400 status code. |
| restriction.py |  The code defines a decorator function `permission_required` that restricts access to a view to users with the admin permission, and another function `admin_required` that applies the `permission_required` decorator to a view. |
| uploadconfig.py |  The code defines a function called allowed_file that checks if a given filename has an allowed extension, which is defined as a set of strings containing png jpg and jpeg |
| __init__.py |  The code defines a Flask application and its various components, including the database, migrations, CSRF protection, login management, and email sending. |

</details>

---

<details><summary>\app\controller</summary>

| File | Summary |
| ---- | ------- |
| AuthController.py |  The code is a Flask-based web application that provides authentication and authorization functionality for users. It includes login, logout, activation, reset password, and send OTP (One-Time Password) features. |
| HomeController.py |  The code defines a Flask application with two routes: index and classify The index route renders an HTML template and passes in a result variable if it exists, as well as a model_exist variable indicating whether the SVM or IndoBERT models exist. The classify route takes a sentence from a form submission, preprocesses it, and then uses the SVM and IndoBERT models to classify the sentiment and intent of the sentence. The results are then stored in a session variable and redirected to the home page. |
| MainController.py |  The code is a Flask application that provides a web interface for analyzing and visualizing comments on YouTube videos. It includes various features such as sentiment analysis, intent classification, and word cloud generation. The primary function of the code is to provide a platform for users to analyze and understand the sentiment and intent behind comments on YouTube videos. |
| UserController.py |  The code is a Python module that defines several functions related to user management, including registering new users, creating new users, updating user profiles, uploading user images, changing user passwords, managing user roles and permissions, and deleting users. |
| WordsController.py |  The code is a Flask web application that provides a platform for users to contribute to a corpus of Indonesian language data. It includes features such as adding, editing, and deleting words, as well as classifying them based on their sentiment and intent. The application also includes a search function and pagination. |

</details>

---

<details><summary>\app\helpers</summary>

| File | Summary |
| ---- | ------- |
| classifying.py |  This code is a Python script that trains and evaluates a Support Vector Machine (SVM) model for sentiment analysis on Indonesian text. The script uses the TF-IDF vectorizer to convert text data into numerical vectors, and then trains an SVM model with the C=1, kernel='linear', and gamma=1 hyperparameters. The model is trained on a dataset of labeled Indonesian text, and the accuracy of the model is evaluated on a test set. The script also includes functions for plotting learning curves and confusion matrices. |
| commons.py |  The code defines a number of functions and classes related to pagination, email confirmation, and database queries. The primary function of the code appears to be to provide a set of tools for working with paginated data, including generating pagination links and filtering results based on search criteria. Additionally, the code includes functionality for sending emails and retrieving data from a database using SQLAlchemy. |
| preprocessing.py |  The code defines a series of functions for text preprocessing, including cleaning, tokenizing, normalizing, stemming, negation handling, and filtering. The primary function of the code is to prepare text data for further analysis in natural language processing tasks. |
| scraping.py |  The code defines a set of functions for interacting with the YouTube API, including retrieving video information, comments, and thumbnails. |
| visualizing.py |  The code defines a function called `generate_wordcloud` that generates a word cloud based on the input text, sentiment, and ID. It also defines a function called `associate_data` that performs association rule mining on the input data using the Apriori algorithm. |

</details>

---

<details><summary>\app\model</summary>

| File | Summary |
| ---- | ------- |
| analysis.py |  The code defines a class called Analysis in the Flask app, which represents a single analysis of a YouTube video. It has several attributes, including an ID, a foreign key to the YouTube video, a user ID, and a timestamp for when the analysis was performed. It also has a relationship with the Comments class and defines a custom string representation. |
| comments.py |  The code defines a Comments class in the Flask app, which represents a comment on an analysis. It has various attributes such as id, analysis_id, title, name, comment, published_at, likes, and replies. The class also has a __repr__ method and a to_print method that returns a dictionary of column names and their corresponding values. |
| compound.py |  The code defines a Compound class in the Flask app, which represents a compound word in the database. It has various attributes and relationships with other models, including Root, User, and Contributor. |
| corpus.py |  The code defines a Corpus class in a Flask application, with properties for text, sentiment, intent, and contributor information, as well as relationships with other classes. |
| informal.py |  The code defines a class Informal that represents a row in a database table, with columns for an informal word, its formal equivalent, the source of the word, the user who contributed it, and the user who edited it. |
| negasi.py |  The code defines a Negasi class in the app module, which represents a negated word or phrase with its source and contributor information. |
| negative.py |  The code defines a Negative model class in Flask, with attributes for a kata negatif, bobot, sumber, kontributor, and editor, as well as a relationship between the model and the User model. |
| positive.py |  The code defines a Positive model class in Flask, which represents a positive sentiment in a database. |
| preprocessed.py |  The code defines a Preprocessed class in the Flask app's database model, with properties for comment ID, casefolded, tokenized, normalized, stemmed, and filtered text, as well as a relationship with the Comments class. |
| processed.py |  The code defines a class Processed that represents a processed comment, with properties for the comment's ID, vectors, SVM classification, IndoBERT classification, intent, and feedback. It also includes a method to print the comment in different formats. |
| root.py |  The code defines a Root class in the app module, which represents a root word in the database. It has various columns for storing data, including an ID, the root word itself, its polarity, and the source of the information. The class also defines relationships with other models, such as Compound and User, and provides a method for printing the data in a specific format. |
| stopword.py |  The code defines a Stopword class in the Flask app, with attributes for a stop word, its source, the user who contributed it, and the user who edited it. It also includes a to_print method to return a dictionary of the class's attributes. |
| user.py |  The code defines a User class in a Flask application, with properties such as username, password, and name, as well as methods for setting and checking passwords, creating an admin user, and defining relationships with other models. |
| youtube.py |  The code defines a class called Youtube that represents a YouTube video, with attributes for the video's ID, title, published date, description, views, likes, favorites, comments, and thumbnail. |
| __init__.py |  The code defines a collection of classes and functions for natural language processing tasks, including text preprocessing, sentiment analysis, and topic modeling. |

</details>

---


<details><summary>\app\routes</summary>

| File | Summary |
| ---- | ------- |
| auth.py |  The code defines a Flask Blueprint named auth with routes for user registration, password reset, account activation, and login/logout functionality. |
| main.py |  The code defines a Flask application with several routes and blueprints, including a main route that handles GET and POST requests, a search route that handles GET and POST requests, and a history route that handles GET and POST requests. The code also defines several functions for handling user feedback, training the model, and generating images. |
| user.py |  The code defines a Flask blueprint named user and its routes, including profil profil/upload-img profil/update-password users user/active/<id>/<is_confirmed> user/admin/<id>/<is_admin> and user/delete/<id> It also imports various functions and classes from other modules, such as Flask's request, render_template, redirect, url_for, session, flask_login's login_required, and app.restriction's admin_required. |
| words.py |  The code defines a Flask blueprint named words and routes for various word-related endpoints, including adding, editing, and deleting words, as well as retrieving a list of words. The routes are protected by login and admin privileges using the Flask-Login and Flask-Admin libraries. |

</details>

---

<details><summary>\app\static\js</summary>

| File | Summary |
| ---- | ------- |
| base.js |  The code is a JavaScript file that contains various functions and event listeners for a web application. The primary function of the code appears to be to initialize various elements on the page, such as dropdown menus, form validation, and modal windows. It also includes functions for handling user input, such as password strength checking and auto-sizing textareas. |
| main.js |  The code is a JavaScript file that defines various functions and event listeners for a website. It appears to be using the Bootstrap framework, as it includes classes such as avbar-mobile and scrolled-offset The code also includes functionality for a preloader, a back-to-top button, and a portfolio slider. Additionally, it includes a type effect and a lightbox for displaying images in a modal window. Overall, the code appears to be responsible for creating a responsive and interactive user experience for a website. |

</details>

---

<details><summary>\app\utils</summary>

| File | Summary |
| ---- | ------- |
| args_helper.py |  The code defines various functions and classes related to data loading, preprocessing, and evaluation for natural language processing tasks. |
| conlleval.py |  The code is a Python implementation of the CoNLL-2000 shared task evaluation script, which evaluates the accuracy of named entity recognition (NER) models on a given dataset. The primary function of the code is to calculate the overall and type-level metrics for the NER model, including precision, recall, and F1 score, as well as the macro-averaged precision, recall, and F1 score for each type of entity. |
| data_utils.py |  This code defines several classes that implement the `Dataset` interface in PyTorch, which is used to load and preprocess data for training machine learning models. The classes are designed to work with different types of datasets, including text classification, sentiment analysis, and aspect-based sentiment analysis.The `AspectExtractionDataset` class loads a dataset of text with aspect-level sentiment annotations and returns a tuple of subwords, sequence labels, and the original sentence. The `NerGritDataset` class loads a dataset of named entities and returns a tuple of subwords, sequence labels, and the original sentence. The `PosTagIdnDataset` class loads a dataset of part-of-speech tags and returns a tuple of subwords, sequence labels, and the original sentence. The `EmotionDetectionDataset` class loads a dataset of emotions and returns a tuple of subwords, sequence labels, and the original sentence. The `EntailmentDataset` class loads a dataset of entailment and returns a tuple of subwords, sequence labels, and the original sentence. The `DocumentSentimentDataset` class loads a dataset of document sentiment and returns a tuple of subwords, sequence labels, and the original sentence. The `KeywordExtractionDataset` class loads a dataset of keyword extraction and returns a tuple of subwords, sequence labels, and the original sentence. The `Q |
| forward_fn.py |  The code defines three forward functions for sequence classification, word classification, and sequence multilabel classification, respectively. Each function takes in a batch of data, prepares the input and label, and then forwards the data through a model to generate predictions and calculate loss. The functions also handle the case where the input data is on GPU. |
| functions.py |  The code defines various functions and classes related to natural language processing (NLP) tasks, including word embedding, tokenization, and model loading. The primary function of the code appears to be loading pre-trained models for NLP tasks, such as sequence classification, token classification, and multi-label classification. |
| metrics.py |  The code defines a set of functions for calculating various metrics for natural language processing tasks, including emotion detection, aspect extraction, named entity recognition, part-of-speech tagging, entailment, document sentiment, keyword extraction, question answering, and news categorization. Each function takes two arguments: a list of predicted labels and a list of true labels, and returns a dictionary of metrics. |

</details>

---


## 🚀 Cara Instalasi

1. Sebelum memulai pemasangan, pastikan Python versi 3.7 ke atas sudah terinstall dan bisa diakses menggunakan terminal.
2. Aktifkan server MySQL, buat database baru dan sesuaikan informasi HOST, USER, PASSWORD, dan NAMA DATABASE pada file .env
3. Buat lingkungan virtual dengan mengetik pada terminal `python -m venv env` 
4. Aktifkan lingkungan virtual dengan mengetik `env\Scripts\activate` (untuk windows) atau `source env/bin/activate` (untuk linux atau macOS)
5. Pasang seluruh dependensi dengan mengetik `pip install -r requirements.txt`, jika Anda menggunakan VGA NVIDIA yang mendukung CUDA lanjutkan ketik `pip install -r requirements2.txt`
6. Lakukan pemasangan aplikasi dengan mengetik `python init.py`
7. Jalankan server flask dengan mengetik `flask run`
8. Buka browser dan arahkan ke url [http://localhost:5000/login](http://localhost:5000/login)
9. Login sebagai superadmin menggunakan username `admin@mail.com` dan password `53cr3tK3y`
10. Struktur menu aplikasi:
```sh
    Beranda
    ├── Tentang
    ├── Latar
    ├── Langkah
    ├── Metode
    ├── Resources
    │   ├── Akar Kata
    │   ├── Kata Berimbuhan
    │   ├── Kata Informal
    │   ├── Kata Negasi
    │   ├── Stopword
    │   ├── Kata Positif
    │   ├── Kata Negatif
    ├── Analisis
    │   ├── Analisis Komentar Youtube
    │   │   └── Hasil Analisis Komentar Video
    │   │       ├── Ringkasan
    │   │       ├── Sebaran Sentimen
    │   │       ├── Hasil Scraping
    │   │       ├── Hasil Casefolding
    │   │       ├── Hasil Tokenisasi
    │   │       ├── Hasil Normalisasi
    │   │       ├── Hasil Stemming
    │   │       ├── Hasil Filtering
    │   │       ├── Hasil Vektorisasi
    │   │       ├── Hasil Klasifikasi
    │   │       └── Awan Kata
    │   ├── Pelatihan Model Analisis
    │   ├── Hasil Pelatihan Sentimen
    │   ├── Hasil Pelatihan Inten
    └── Login
        ├── User
        │   ├── Profil
        │   ├── Logout
        └── Admin
            ├── Kelola Pengguna
            ├── Profil
            └── Logout
```
------------------------------------------
