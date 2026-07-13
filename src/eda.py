import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("data/cleaned_returns.csv")

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

product_counts = df["Product"].value_counts()

print(product_counts)
product_counts.plot(kind="bar")

plt.title("Number of Returns by Product")
plt.xlabel("Product")
plt.ylabel("Number of Returns")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

category_counts = df["Category"].value_counts()

print(category_counts)

category_counts.plot(kind="bar")

plt.title("Return Categories")

plt.xlabel("Category")

plt.ylabel("Count")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()

all_words = " ".join(df["Cleaned_Reason"])

wordcloud = WordCloud(
    width=900,
    height=500,
    background_color="white"
).generate(all_words)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Most Common Words in Return Reasons")

plt.show()

from collections import Counter

words = all_words.split()

word_freq = Counter(words)

print(word_freq.most_common(10))