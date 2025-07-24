import pickle

# Only 2 classes since you removed neutral
labels = ["negative", "positive"]

# Save as label_encoder.pkl
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(labels, f)

print("label_encoder.pkl updated!")
