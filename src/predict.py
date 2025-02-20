import joblib
import pandas as pd
from preprocess import preprocess_text

def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_spam(model, vectorizer, message):
    cleaned_message = preprocess_text(message)
    message_vectorized = vectorizer.transform([cleaned_message])
    prediction = model.predict(message_vectorized)
    return "Spam" if prediction[0] == 1 else "Ham"

if __name__ == "__main__":
    model_path = './models/spam_detection_model.pkl'
    vectorizer_path = './models/tfidf_vectorizer.pkl'
    
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    
    # Example usage
    new_message = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim."
    result = predict_spam(model, vectorizer, new_message)
    print(result)