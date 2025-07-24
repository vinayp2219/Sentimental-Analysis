import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load data
train_df = pd.read_csv("train.csv")  # Should have: id, tweet, label
test_df = pd.read_csv("test.csv")    # Should have: id, tweet

# Convert labels: 0 → -1 (negative), 1 → 1 (neutral), 2 → 1 (positive)
label_map = {0: -1, 1: 1, 2: 1}
train_df['label'] = train_df['label'].map(label_map)

# Features and labels
X = train_df['tweet']
y = train_df['label']

# Pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X, y)

# Predict on test data
test_df['label'] = model.predict(test_df['tweet'])

# Save results
test_df.to_csv("test_pos_neutral_as_positive.csv", index=False)
train_df.to_csv("train_pos_neutral_as_positive.csv", index=False)

# Optional evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print(classification_report(
    y_val, y_pred,
    labels=[-1, 1],
    target_names=["Negative (-1)", "Positive+Neutral (1)"]
))
