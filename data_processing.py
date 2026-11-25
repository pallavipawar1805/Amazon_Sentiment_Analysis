
import pandas as pd
import re

def process_data():

    # Step 1: Load dataset
    df = pd.read_csv("Amazon_Reviews.csv")

    # Step 2: Create sentiment function
    def get_sentiment(score):
        if score <= 2:
            return 'Negative'
        elif score == 3:
            return 'Neutral'
        else:
            return 'Positive'

    df['Sentiment'] = df['Score'].apply(get_sentiment)

    # Step 3: Handle missing text
    df['Text'] = df['Text'].fillna('')

    # Step 4: Text cleaning function
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Step 5: Apply text cleaning
    df['Cleaned_Text'] = df['Text'].apply(clean_text)

    # Step 6: Return processed DataFrame
    return df


if __name__ == "__main__":
    df = process_data()
    print(df[['Score', 'Sentiment', 'Cleaned_Text']].head())
