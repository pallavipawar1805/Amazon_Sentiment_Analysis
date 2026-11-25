from fastapi import FastAPI
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI(title="Amazon Sentiment Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("sentiment_model.pkl")

# Load TF-IDF vectorizer                    
tfidf = joblib.load("tfidf_vectorizer.pkl")

print("Model and Vectorizer Loaded Successfully!")

from pydantic import BaseModel

# Define input structure
class ReviewInput(BaseModel):
    review: str

# Create prediction endpoint
@app.post("/predict")
def predict_sentiment(data: ReviewInput):

    # Get review text
    review_text = [data.review]

    # Convert text into TF-IDF features
    review_tfidf = tfidf.transform(review_text)

    # Predict sentiment
    prediction = model.predict(review_tfidf)

    return {
        "review": data.review,
        "predicted_sentiment": prediction[0]
    }