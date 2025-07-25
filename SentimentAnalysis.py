import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import nltk
import re

# Optional: only needed once
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Tokenizer
def tokenize(text):
    return word_tokenize(text)

# Load training data
train_df = pd.read_csv("train5(1).csv")
train_df = train_df.dropna(subset=['tweet', 'label'])
train_df['label'] = train_df['label'].astype(int)
X = train_df['tweet']
y = train_df['label']

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred, labels=[-1, 1], target_names=['Negative', 'Positive']))

# Load test data
test_df = pd.read_csv("test5.csv")
test_df = test_df.dropna(subset=['tweet'])

# Predict on test data
test_df['predicted_label'] = model.predict(test_df['tweet'])

# Save to file
test_df.to_csv("test_with_predictions.csv", index=False)
print("✅ Test predictions saved to test_with_predictions.csv")

print(train_df['label'].value_counts())
