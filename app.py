from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type:ignore
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("sentiment_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder classes (just the list of classes)
with open("label_encoder.pkl", "rb") as f:
    label_classes = pickle.load(f)

# Convert index to label
index_to_label = dict(enumerate(label_classes.classes_))

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    probabilities = None

    if request.method == "POST":
        text = request.form.get("text")
        if text:
            # Preprocess
            sequence = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=200)

            # Predict
            # Load the actual LabelEncoder object
            with open("label_encoder.pkl", "rb") as f:
                label_encoder = pickle.load(f)

# Predict sentiment
            probs = model.predict(padded)[0]
            predicted_index = int(np.argmax(probs))
            sentiment = label_encoder.inverse_transform([predicted_index])[0]
            probabilities = {label_encoder.inverse_transform([i])[0]: f"{p*100:.2f}%" for i, p in enumerate(probs)}


    return render_template("index.html", sentiment=sentiment, probabilities=probabilities)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
