import joblib

# 1. Učitavanje modela i vektorizatora
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

print("Unesi poruku za klasifikaciju (ili 'exit' za izlaz):")

while True:
    text = input("> ")
    if text.lower() == 'exit':
        break
    
    # 2. Transformacija unetog teksta
    text_tfidf = vectorizer.transform([text])
    
    # 3. Predikcija
    prediction = model.predict(text_tfidf)[0]
    print("Predikcija:", prediction)