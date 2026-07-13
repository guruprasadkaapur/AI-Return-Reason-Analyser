import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_returns.csv")

# Features and Target
X = df["Cleaned_Reason"]
y = df["Category"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

# Train Logistic Regression Model
model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("=" * 50)

# Classification Report
print("\nClassification Report\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the TF-IDF vectorizer
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model saved to models/model.pkl")
print("✅ Vectorizer saved to models/vectorizer.pkl")