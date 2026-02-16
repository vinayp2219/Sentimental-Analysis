import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense #type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Load dataset
df = pd.read_csv("latest.csv")

texts = df["text"].astype(str)
labels = df["sentiment"]

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=200, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=200, padding="post")

# Build Model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=200),
    LSTM(64),
    Dense(len(np.unique(y_train)), activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train model
model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    validation_split=0.1,
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)

# Predictions
y_pred_probs = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy (Sklearn)
acc = accuracy_score(y_test, y_pred)

# F1 Scores
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

# Save model
model.save("sentiment_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
