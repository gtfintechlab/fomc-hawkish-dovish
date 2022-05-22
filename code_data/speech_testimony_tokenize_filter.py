import os

import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import re

A1 = ["inflation expectation", "interest rate", "bank rate", "fund rate", "price", "economic activity", "inflation",
      "employment"]
B1 = ["unemployment", "growth", "exchange rate", "productivity", "deficit", "demand", "job market", "monetary policy"]

# a list of select words that are used to filter FOMC sentences. We build dictionary based on these sentences
def sentence_tokenize(text):
    return sent_tokenize(text.replace("\n", ""))

def save_csv_filtered(token_list, name, output_directory):
    new_name = name.replace(".txt", "") + "_filtered"
    df = pd.DataFrame(token_list, columns=["sentence"])
    df.to_csv(output_directory + new_name + ".csv")

def dictionary_filter(token_list):
    temp = []
    for sentence in token_list:
        sentence = sentence.lower()
        if any(word1 in sentence for word1 in A1) or any(word2 in sentence for word2 in B1):
            temp.append(sentence)
    return temp



def execute(directory, output_directory):
    for f_name in os.listdir(directory):
        file_path = os.path.join(directory, f_name)
        if os.path.isfile(file_path) and not f_name.startswith("."):
            print(f_name)
            with open(file_path, errors='ignore') as f:
                content = f.readlines()
            sent_tokens = [x.strip() for x in content]
            #content = [x[:512] for x in content]

            sent_tokens = dictionary_filter(sent_tokens)  # filter the sentences based on ones that contain "select words"

            save_csv_filtered(sent_tokens, f_name, output_directory)  # store the filtered sentences into csv files


if __name__ == "__main__":
    execute("../data/speech_testimony_text_data/speech/", "./speech-filtered/")
    execute("../data/speech_testimony_text_data/testimony/", "./testimony-filtered/")