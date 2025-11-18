import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Učitavanje podataka
df = pd.read_csv('products.csv')
X = df['Product Title']
y = df['Category Label']

# 2. Podela na trening i test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Transformacija teksta u numerički oblik
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Treniranje modela
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Evaluacija
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Čuvanje modela i vektorizatora
joblib.dump(model, 'models/spam_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("Model i vektorizator su sačuvani u folderu 'models/'.")
