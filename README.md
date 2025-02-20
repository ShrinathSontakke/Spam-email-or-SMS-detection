# Spam Email Detection

This project implements a spam detection model using the SMS Spam Collection Dataset. The model is built using machine learning techniques and is designed to classify SMS messages as either "spam" or "ham" (not spam).

## Project Structure

```
spam-email-detection
├── data
│   └── spam.csv               # Contains the SMS Spam Collection Dataset
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

1. Clone the repository:

   ```
   git clone <repository-url>
   cd spam-email-detection
   ```

2. Install the required libraries:

   ```
   pip install -r requirements.txt
   ```

3. Download the SMS Spam Collection Dataset from Kaggle and place it in the `data` directory as `spam.csv`.

## Usage

1. Open the Jupyter notebook `notebooks/spam_detection.ipynb` to follow the complete workflow for building the spam detection model.

2. To preprocess the data, run the functions in `src/preprocess.py`.

3. Train the model by executing the code in `src/train.py`.

4. Make predictions on new SMS messages using the functions in `src/predict.py`.
