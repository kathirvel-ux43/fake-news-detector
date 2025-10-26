import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import messagebox

MODEL_FILE = "fake_news_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

# --- Load or Train Model ---
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    print(" Model and vectorizer loaded successfully!")
else:
    print(" Training model...")
    df = pd.read_csv("fake_news.csv")

    X = df['Title']
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # --- Evaluate model ---
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n Model Accuracy: {accuracy * 100:.2f}%")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save for future use
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print("\n Model trained and saved successfully!")

# --- GUI Function ---
def check_headline():
    headline = entry.get()
    if headline.strip() == "":
        messagebox.showwarning("Input Error", "Please enter a headline")
        return
    tfidf_headline = vectorizer.transform([headline])
    prediction = model.predict(tfidf_headline)[0]
    messagebox.showinfo("Prediction", f"Headline:\n{headline}\n\nPrediction: {prediction}")

# --- GUI Setup ---
root = tk.Tk()
root.title(" Fake News Detector")
root.geometry("500x200")

label = tk.Label(root, text="Enter a news headline:", font=("Arial", 14))
label.pack(pady=10)

entry = tk.Entry(root, width=60, font=("Arial", 12))
entry.pack(pady=5)

button = tk.Button(root, text="Check", command=check_headline, font=("Arial", 12))
button.pack(pady=10)

root.mainloop()
