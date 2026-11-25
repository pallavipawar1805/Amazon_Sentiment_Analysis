# Step 1: Import library
import pandas as pd

# Step 2: Load dataset
df = pd.read_csv("Amazon_Reviews.csv")

# Step 3: Check dataset dimensions
print("Dataset Shape:", df.shape)

# Step 4: Display first 5 records
print(df.head())

# Step 5: Display column names
print(df.columns)

# Step 2: Understand Summary column

print("\nFirst 10 Summary values:")

print(df["Summary"].head(10))

# Step 3: Check missing values in Summary

print("\nMissing values in Summary:")

print(df["Summary"].isnull().sum())

# Step 4: Check duplicate summaries

print("\nDuplicate Summary count:")

print(df["Summary"].duplicated().sum())

# Step 5: Understand Score column

print("\nUnique Score values:")

print(df["Score"].unique())

# Step 6: score distribution
print("\nScore Distribution:")
print(df["Score"].value_counts().sort_index())

(df['Score'].value_counts(normalize=True) * 100).round(2)