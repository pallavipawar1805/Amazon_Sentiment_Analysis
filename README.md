# 🛒 Amazon Review Sentiment Analysis

## 📌 Project Overview
This project analyzes Amazon product reviews and predicts the sentiment of the review as:

- Positive 😊  
- Negative 😞  
- Neutral 😐  

The model is trained using Natural Language Processing (NLP) techniques and deployed using a Streamlit web application.

---

## 🚀 Features
- Text preprocessing (lowercasing, removing special characters)
- Stopwords removal using NLTK
- TF-IDF vectorization
- Multi-class sentiment classification (Positive, Negative, Neutral)
- Model comparison:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (LinearSVC)
- Final model selected: **LinearSVC**
- Interactive web app using Streamlit

---

## 📂 Project Structure
amazon-sentiment-analysis/
│
├── data/                          # Dataset folder (optional but recommended)
│   └── Amazon_Reviews.csv
│
├── notebook/
│   └── amazon_sentiment_analysis.ipynb   # Your Colab notebook
│
├── model/                         # Saved ML artifacts
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py                         # Streamlit application
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Ignore unnecessary files
