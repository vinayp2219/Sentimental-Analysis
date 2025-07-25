# import pandas as pd
# import re

# # Load the dataset
# df = pd.read_csv("train_pos_neutral_as_positive.csv")

# # Cleaning function
# def clean_tweet(text):
#     text = text.lower()                                  # lowercase
#     text = re.sub(r'@\w+', '', text)                     # remove @mentions
#     text = re.sub(r'#\w+', '', text)                     # remove hashtags
#     text = re.sub(r'http\S+', '', text)                  # remove URLs
#     text = re.sub(r'[^a-z\s]', '', text)                 # remove punctuation, numbers, emojis
#     text = re.sub(r'\s+', ' ', text).strip()             # remove extra whitespace
#     return text

# # Apply cleaning
# df['tweet'] = df['tweet'].astype(str).apply(clean_tweet)

# # Optionally drop tweets that are too short (less than 3 words)
# df = df[df['tweet'].apply(lambda x: len(x.split()) >= 3)]

# # Save cleaned file
# df.to_csv("train_clean_final.csv", index=False)
# print("✅ Data cleaned and saved to train_clean_final.csv")


import pickle

# Open tokenizer.pkl in binary read mode
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
