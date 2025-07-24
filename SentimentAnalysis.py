# import pandas as pd
# import pickle
# from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
# from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
# from tensorflow.keras.models import Sequential #type: ignore
# from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding #type: ignore

# # Load dataset
# df = pd.read_csv("./Tweets.csv")

# # Keep all 3 sentiments
# tweet_df = df[['text', 'airline_sentiment']]

# # Factorize labels (0, 1, 2)
# sentiment_label = tweet_df.airline_sentiment.factorize()
# y = sentiment_label[0]

# # Tokenize text
# tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(tweet_df.text)
# X = tokenizer.texts_to_sequences(tweet_df.text)
# X = pad_sequences(X, maxlen=200)
# vocab_size = len(tokenizer.word_index) + 1

# # Build model
# model = Sequential()
# model.add(Embedding(vocab_size, 32, input_length=200))
# model.add(SpatialDropout1D(0.25))
# model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))

# # Compile model
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train model
# model.fit(X, y, validation_split=0.2, epochs=7, batch_size=32)

# # Save everything
# model.save("sentiment_model.h5")

# with open("tokenizer.pkl", "wb") as f:
#     pickle.dump(tokenizer, f)

# with open("label_encoder.pkl", "wb") as f:
#     pickle.dump(sentiment_label[1], f)


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
train_df = pd.read_csv("train5.csv")
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
