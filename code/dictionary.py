import os

import pandas as pd
from nltk.tokenize import sent_tokenize
import re

total_lines = 0
num_words = 0
filtered_lines = 0
filtered_words = 0

A1 = ["inflation expectation", "interest rate", "bank rate", "fund rate", "price", "economic activity", "inflation",
      "employment"]
B1 = ["unemployment", "growth", "exchange rate", "productivity", "deficit", "demand", "job market", "monetary policy"]


# a list of select words that are used to filter FOMC sentences. We build dictionary based on these sentences

def sentence_tokenize(text):
    global total_lines
    sent_tokens = sent_tokenize(text.replace("\n", ""))
    total_lines += len(sent_tokens)
    return sent_tokens


def save_csv(token_list, name):
    new_name = name.replace(".pdf", "") + "_filtered"
    df = pd.DataFrame(token_list, columns=["sentence"])
    df.to_csv('/Users/suvanpaturi/Documents/Meeting-Minutes-Filtered/' + new_name + ".csv")


def dictionary_filter(token_list):
    temp = []
    for sentence in token_list:
        if any(word1 in sentence for word1 in A1) or any(word2 in sentence for word2 in B1):
            temp.append(sentence)
    return temp


directory = "/Users/suvanpaturi/Documents/Meeting-Minutes"
for f_name in os.listdir(directory):
    file_path = os.path.join(directory, f_name)
    if os.path.isfile(file_path) and not f_name.startswith("."):
        print(f_name)
        file = open(str(file_path), 'r')  # open text file to be read
        text = file.read()
        text = re.sub(r'(?<=[.,;])(?=[^\s])', r' ', text)

        sent_tokens = sentence_tokenize(text)  # tokenize the text in terms of sentences
        sent_tokens = [re.sub(r'\s+', ' ', sent) for sent in sent_tokens]
        sent_tokens = dictionary_filter(sent_tokens)  # filter the sentences based on ones that contain "select words"
        print(sent_tokens)
        save_csv(sent_tokens, f_name)  # store the filtered sentences into csv files

        print(len(sent_tokens))
        filtered_lines += len(sent_tokens)

print(total_lines)
print(total_lines / 200)
print("#---------------------#")
print(filtered_lines)
print(filtered_lines / 200)


