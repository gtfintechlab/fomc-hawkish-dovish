import os

import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
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
    return sent_tokenize(text.replace("\n", ""))


def save_csv_filtered(token_list, name):
    new_name = name.replace(".txt", "") + "_filtered"
    df = pd.DataFrame(token_list, columns=["sentence"])
    df.to_csv('./Meeting-Minutes-Filtered/' + new_name + ".csv")

def save_csv(token_list, name):
    new_name = name.replace(".txt", "") + "_tokenized"
    df = pd.DataFrame(token_list, columns=["sentence"])
    df.to_csv('./Meeting-Minutes-Tokenized/' + new_name + ".csv")

def dictionary_filter(token_list):
    temp = []
    for sentence in token_list:
        if any(word1 in sentence for word1 in A1) or any(word2 in sentence for word2 in B1):
            temp.append(sentence)
    return temp



directory = "./Meeting-Minutes"
for f_name in os.listdir(directory):
    file_path = os.path.join(directory, f_name)
    if os.path.isfile(file_path) and not f_name.startswith("."):
        print(f_name)
        file = open(str(file_path), 'r', encoding="utf-8")  # open text file to be read
        text = file.read()
        text = re.sub(r'(?<=[.,;])(?=[^\s])', r' ', text)
        sent_tokens = sentence_tokenize(text)  # tokenize the text in terms of sentences
        sent_tokens = [re.sub(r'\s+', ' ', sent) for sent in sent_tokens]

        total_lines += len(sent_tokens)  # count number of sentences before filter
        for sent in sent_tokens:
            num_words += len(word_tokenize(sent))  # count number of words before filter
        
        save_csv(sent_tokens, f_name)

        sent_tokens = dictionary_filter(sent_tokens)  # filter the sentences based on ones that contain "select words"
        #print(sent_tokens)
        save_csv_filtered(sent_tokens, f_name)  # store the filtered sentences into csv files

        filtered_lines += len(sent_tokens)  # number of filtered sentences
        for sent in sent_tokens:
            filtered_words += len(word_tokenize(sent))  # count number of words before filter

print(total_lines)
print(num_words)
print("#---------------------#")
print(filtered_lines)
print(filtered_words)
