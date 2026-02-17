import pandas as pd
import numpy as np
import pickle
import re

from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout #type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("latest.csv", encoding="latin1")

texts = df["text"].astype(str)
labels = df["sentiment"]

# -----------------------------
# Clean Text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

texts = texts.apply(clean_text)

# -----------------------------
# Encode Labels
# -----------------------------
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

print("Class Distribution:")
print(df["sentiment"].value_counts())
print("Encoded Classes:", le.classes_)

# -----------------------------
# Train Test Split (Stratified)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

# -----------------------------
# Tokenization
# -----------------------------
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=200, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=200, padding="post")

# -----------------------------
# Improved Model
# -----------------------------
model = Sequential([
    Embedding(10000, 128, input_length=200),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(np.unique(y_train)), activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    X_train_pad,
    y_train,
    epochs=12,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# -----------------------------
# Evaluate
# -----------------------------
loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)

y_pred_probs = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print("\nModel Evaluation Results")
print("-------------------------")
print(f"Test Accuracy (Keras): {accuracy * 100:.2f}%")
print(f"Accuracy (Sklearn): {acc * 100:.2f}%")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -----------------------------
# Save Model & Objects
# -----------------------------
model.save("sentiment_model.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
