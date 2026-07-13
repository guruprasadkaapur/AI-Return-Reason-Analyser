# 📦 AI Return Analytics Dashboard

An AI-powered NLP application that automatically analyzes e-commerce return reasons and classifies them into business-friendly categories. The project includes a Machine Learning model, bulk CSV processing, and an interactive Streamlit dashboard for visualization and insights.

---

## 🚀 Project Overview

E-commerce companies receive thousands of product returns every day.

Customers often provide return reasons in free text such as:

* "The shirt is too small."
* "Battery drains within two hours."
* "Wrong color received."
* "Product arrived damaged."

Manually analyzing these return reasons is time-consuming.

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to automatically classify return reasons into predefined categories and provide business insights through an interactive dashboard.

---

## ✨ Features

* 📂 Bulk CSV Upload
* 🤖 AI-based Return Reason Classification
* 📊 Interactive Dashboard
* 📈 Product-wise Return Analysis
* 📋 Category Summary
* 📥 Download Prediction Results
* ⚡ Fast TF-IDF Text Processing
* 🎯 Logistic Regression Classification Model

---

## 🛠️ Tech Stack

### Programming

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* WordCloud
* Streamlit

### Machine Learning

* TF-IDF Vectorizer
* Logistic Regression

---

## 🧠 Machine Learning Pipeline

1. Data Collection
2. Text Cleaning
3. Tokenization
4. Stopword Removal
5. Lemmatization
6. TF-IDF Vectorization
7. Train/Test Split
8. Logistic Regression Model
9. Prediction
10. Dashboard Visualization

---

## 📊 Dashboard Features

* Upload CSV
* Predict return categories
* KPI Cards
* Category Distribution
* Product-wise Analysis
* Download Processed CSV

---

## ▶️ How to Run

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Return-Reason-Analyser.git
```

### Navigate to the project

```bash
cd AI-Return-Reason-Analyser
```

### Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python src/train.py
```

### Launch the dashboard

```bash
streamlit run app.py
```

---

## 📸 Dashboard Preview


```
images/screenshot_1.png
images/screenshot_2.png
images/screenshot_3.png

```

---

## 📈 Future Enhancements

* Product & Category Filters
* Confidence Score Display
* WordCloud Visualization
* Pie Charts
* AI-generated Business Recommendations
* Excel & PDF Export
* Advanced Analytics Dashboard

---

## 👨‍💻 Author

**Guru Prasad**


---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
