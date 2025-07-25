import pickle

# Open tokenizer.pkl in binary read mode
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
