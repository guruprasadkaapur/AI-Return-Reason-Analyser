import pickle

# Load trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("=" * 50)
print("AI Return Reason Predictor")
print("=" * 50)

while True:
    # User input
    reason = input("\nEnter a return reason (or type 'exit' to quit): ")

    if reason.lower() == "exit":
        print("Goodbye!")
        break

    # Convert text into TF-IDF features
    reason_vector = vectorizer.transform([reason.lower()])

    # Predict category
    prediction = model.predict(reason_vector)

    print("\nPredicted Category:", prediction[0])