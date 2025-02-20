# Spam Email Detection

This project implements a spam detection model using the SMS Spam Collection Dataset. The model is built using machine learning techniques and is designed to classify SMS messages as either "spam" or "ham" (not spam).

## Project Structure

```
spam-email-detection
├── data
│   └── spam.csv               # Contains the SMS Spam Collection Dataset
├── models
│   └── spam_detection_model.pkl  # Trained spam detection model
│   └── tfidf_vectorizer.pkl      # Trained TF-IDF vectorizer
├── notebooks
│   └── spam_detection.ipynb    # Jupyter notebook for the complete workflow
├── src
│   ├── preprocess.py           # Functions for data cleaning and preprocessing
│   ├── train.py                # Logic for training the spam detection model
│   └── predict.py              # Logic for making predictions on new messages
├── requirements.txt            # Required Python libraries
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**

   ```sh
   git clone <repository-url>
   cd spam-email-detection
   ```

2. **Create and activate a virtual environment:**

   - On Windows:
     ```sh
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     python -m venv venv
     source venv/bin/activate
     ```

3. **Install the required libraries:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Download the SMS Spam Collection Dataset from Kaggle and place it in the `data` directory as `spam.csv`.**

## Usage

1. **Open the Jupyter notebook** `notebooks/spam_detection.ipynb` to follow the complete workflow for building the spam detection model.

2. **Preprocess the data** by running the functions in `src/preprocess.py`.

3. **Train the model** by executing the code in `src/train.py`:

   ```sh
   python src/train.py
   ```

4. **Make predictions** on new SMS messages using the functions in `src/predict.py`:

   ```sh
   python src/predict.py
   ```

## Example

To predict whether a new message is spam or ham, you can modify the `new_message` variable in `src/predict.py` and run the script:

```python
if __name__ == "__main__":
    model_path = './models/spam_detection_model.pkl'
    vectorizer_path = './models/tfidf_vectorizer.pkl'

    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

    # Example usage
    new_message = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim."
    result = predict_spam(model, vectorizer, new_message)
    print(result)
```

## Results

After running `predict.py`, you should see an output like:

```
Spam
```

This indicates that the example message provided in the script is classified as spam.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome all contributions!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
