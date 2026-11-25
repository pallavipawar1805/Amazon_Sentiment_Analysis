# Amazon Sentiment Analysis

## Project Overview
An end-to-end Machine Learning application that analyzes Amazon product reviews and classifies them into three sentiment categories: **Positive, Negative, and Neutral**.

The application uses a trained ML model with a FastAPI backend and React frontend to provide sentiment predictions.

## 1. Project Objectives
- Automate sentiment classification of Amazon product reviews.
- Understand customer opinions and feedback.
- Build an interactive application for real-time sentiment prediction.

## 2. Technologies Used
- **Programming:** Python, JavaScript
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Backend:** FastAPI
- **Frontend:** React.js, Vite, CSS
- **API Communication:** REST API

## 3. Project Structure

```text
Amazon_Sentiment_Analysis/
│
├── app.py
├── data_analysis.py
├── data_processing.py
├── model_training.py
├── model_evaluation.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── package-lock.json
    └── src/
        ├── App.jsx
        ├── App.css
        ├── index.css
        └── main.jsx
```

## 4. Project Workflow

1. **Data Analysis:** Explored the dataset and analyzed sentiment distribution.
2. **Data Processing:** Handled missing values, duplicates, and text preprocessing.
3. **Feature Extraction:** Converted review text into numerical features using a Vectorizer.
4. **Model Training:** Trained a Machine Learning model for sentiment classification.
5. **Model Evaluation:** Evaluated performance using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
6. **Backend Development:** Developed a FastAPI endpoint for sentiment prediction.
7. **Frontend Development:** Created an interactive React interface for entering reviews and displaying predictions.

## 5. Model Performance

The model was evaluated on 113,691 reviews.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 79.56% |
| Weighted Precision | 87.37% |
| Weighted Recall | 79.56% |
| Weighted F1-score | 82.31% |

### Class-wise Performance

| Sentiment | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.65 | 0.75 | 0.70 |
| Neutral | 0.28 | 0.66 | 0.39 |
| Positive | 0.97 | 0.82 | 0.89 |

## 6. Application Features

- Accepts customer product reviews as input.
- Predicts Positive, Negative, or Neutral sentiment.
- Provides real-time predictions through a FastAPI backend.
- Interactive React-based user interface.
- Displays prediction results clearly.

## 7. Installation and Setup

### Backend Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Amazon_Sentiment_Analysis.git
```

Navigate to the project directory:

```bash
cd Amazon_Sentiment_Analysis
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Backend API:

```text
http://127.0.0.1:8000
```

### Frontend Setup

Navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the React application:

```bash
npm run dev
```

Open the local URL displayed by Vite in your browser.

## 8. API Endpoint

**POST /predict**

Accepts a product review and returns its predicted sentiment.

Example request:

```json
{
  "review": "This product is amazing and has excellent quality"
}
```

Example response:

```json
{
  "predicted_sentiment": "Positive"
}
```

## 9. Future Improvements

- Improve Neutral sentiment classification.
- Experiment with advanced NLP techniques.
- Improve model performance through hyperparameter tuning.
- Deploy the application for public access.



