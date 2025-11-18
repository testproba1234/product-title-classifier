Spam Classifier

Ovaj projekat je jednostavan klasifikator poruka (spam/ham) napravljen u Pythonu koristeći scikit-learn. Sadrži dve glavne skripte:

train_model.py – za treniranje modela

predict_category.py – za interaktivno testiranje novih poruka

Instalacija:

Struktura projekta:

data/

products.csv

colab/

best_model.pkl (sačuvani model)

tfidf_vectorizer.pkl (sačuvani TF-IDF vektorizator)

data_analysis_and_cleaning.ipynb (analiza i čišćenje strukture podataka)

model_training.ipynb (treniranje modela)

model/

train_model.py (trenirani model)

predict_category.py (interaktivno testiranje)

README.md

Upotreba:

Treniranje modela:
Pokreni python train_model.py
Skripta će trenirati model, prikazati tačnost i sačuvati model i vektorizator u folderu models/.

Interaktivno testiranje:
Pokreni python predict_category.py
Unesi poruku i dobijaš predikciju (spam ili ham).
Unesi 'exit' za izlaz iz skripte.

Napomene:

Dataset je očekivan u CSV formatu sa kolonama text i label.
Model koristi MultinomialNB sa TF-IDF vektorizacijom.

Projekat se može proširiti dodavanjem drugih modela ili unapređenjem predprocesiranja teksta.
