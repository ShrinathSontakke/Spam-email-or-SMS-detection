import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords if not available
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv('./data/spam.csv', encoding='latin-1')

# Print the column names for debugging
print("Columns in dataset:", data.columns)

# Select relevant columns based on actual column names
data = data[['label', 'text']]  # Adjust column names as per actual dataset
data.columns = ['label', 'message']  # Renaming columns for clarity

# Drop any missing values
data = data.dropna()

# Preprocess the text data
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters & numbers
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

# Apply preprocessing
data.loc[:, 'cleaned_message'] = data['message'].apply(preprocess_text)

# Save the cleaned data
data.to_csv('./data/cleaned_spam.csv', index=False)

print("Preprocessing complete! Cleaned data saved to './data/cleaned_spam.csv'.")
