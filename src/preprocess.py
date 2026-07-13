import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run once)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load dataset
df = pd.read_csv("data/returns.csv")

print("Original Dataset:")
print(df.head())

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

# Apply cleaning
df["Cleaned_Reason"] = df["Return_Reason"].apply(clean_text)

# Display results
print("\nCleaned Dataset:")
print(df[["Return_Reason", "Cleaned_Reason"]])

# Save cleaned dataset
df.to_csv("data/cleaned_returns.csv", index=False)

print("\n✅ Cleaned dataset saved as data/cleaned_returns.csv")