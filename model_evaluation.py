
# Step 1: Import libraries

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from data_processing import process_data


# Step 2: Load processed data

df = process_data()

X = df['Cleaned_Text']
y = df['Sentiment']


# Step 3: Recreate the same train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Step 4: Load saved model and TF-IDF vectorizer

model = joblib.load("sentiment_model.pkl")

tfidf = joblib.load("tfidf_vectorizer.pkl")

print("Model and Vectorizer Loaded Successfully!")


# Step 5: Transform test data using saved vectorizer

X_test_tfidf = tfidf.transform(X_test)


# Step 6: Predict sentiment

y_pred = model.predict(X_test_tfidf)


# Step 7: Calculate evaluation metrics

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average='weighted', zero_division=0
)

recall = recall_score(
    y_test, y_pred, average='weighted', zero_division=0
)

f1 = f1_score(
    y_test, y_pred, average='weighted', zero_division=0
)


# Step 8: Display evaluation results

print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# Step 9: Classification Report

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        labels=['Negative', 'Neutral', 'Positive'],
        zero_division=0
    )
)


# Step 10: Confusion Matrix

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=['Negative', 'Neutral', 'Positive']
)

cm_df = pd.DataFrame(
    cm,
    index=['Actual Negative', 'Actual Neutral', 'Actual Positive'],
    columns=['Predicted Negative', 'Predicted Neutral', 'Predicted Positive']
)

print(cm_df)

print("\nModel Evaluation Completed!")

# Step 11: Analyze incorrectly predicted Neutral reviews

results = pd.DataFrame({
    'Review': X_test,
    'Actual': y_test,
    'Predicted': y_pred
})

neutral_errors = results[
    (results['Actual'] == 'Neutral') &
    (results['Predicted'] != 'Neutral')
]

print("\nIncorrectly Predicted Neutral Reviews:")
print(neutral_errors[['Review', 'Actual', 'Predicted']].head(10))

print("\nTotal Neutral Errors:", len(neutral_errors))