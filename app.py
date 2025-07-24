from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("sentiment_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder (as a dictionary)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Reverse the label encoder if needed (index to label)
index_to_label = dict(enumerate(label_encoder))

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
            probs = model.predict(padded)[0]
            predicted_index = int(np.argmax(probs))
            sentiment = index_to_label[predicted_index]
            probabilities = {index_to_label[i]: f"{p*100:.2f}%" for i, p in enumerate(probs)}

    return render_template("index.html", sentiment=sentiment, probabilities=probabilities)

if __name__ == "__main__":
    app.run(debug=True)
