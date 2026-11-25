
from data_processing import process_data
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# Step 1: Get processed data
df = process_data()

# Step 2: Define Input and Target
X = df['Cleaned_Text']
y = df['Sentiment']

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 4: Initialize TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95,
    dtype=np.float32,
    sublinear_tf=True,
    ngram_range=(1, 1)
)

# Step 5: Fit only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Step 6: Transform test data
X_test_tfidf = tfidf.transform(X_test)

# Step 7: Check shapes
print("Training data:", X_train_tfidf.shape)
print("Testing data:", X_test_tfidf.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

#step 8:import model
from sklearn.linear_model import LogisticRegression

# Step 9: Initialize Model
model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced'
)

# Step 10: Train Model
print("Starting Model Training...")

model.fit(X_train_tfidf, y_train)

print("Model Training Completed!")

# Step 11: Save trained model and vectorizer

os.makedirs("models", exist_ok=True)

import joblib

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Model and TF-IDF Vectorizer Saved Successfully!")

print("Training Class Distribution:")
print(y_train.value_counts())