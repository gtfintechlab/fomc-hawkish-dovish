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


def save_csv(token_list, name, location):
    new_name = name.replace(".pdf", "").replace(".csv", "").replace(".txt", "") + "_filtered"
    df = pd.DataFrame(token_list, columns=["sentence"])
    df.to_csv(location + new_name + ".csv")


def dictionary_filter(token_list):  # need to see if mm and pc follow this, speech is valid
    temp = []
    for sentence in token_list:
        if any(word1 in sentence.lower() for word1 in A1) or any(word2 in sentence.lower() for word2 in B1):
            temp.append(sentence)
    return temp


def old_mm_filter(token_list):  # need to see if mm and pc follow this, speech is valid
    temp = []
    for sentence in token_list:
        if any(word1 in sentence for word1 in A1) or any(word2 in sentence for word2 in B1):
            temp.append(sentence)
    return temp


def mm_call(input_path, output_path):

    total_lines = 0
    total_words = 0
    filtered_lines = 0
    filtered_words = 0

    for f_name in os.listdir(input_path):
        file_path = os.path.join(input_path, f_name)
        if os.path.isfile(file_path) and not f_name.startswith("."):
            print(f_name)
            file = open(str(file_path), 'r')  # open text file to be read
            text = file.read()
            text = re.sub(r'(?<=[.,;])(?=[^\s])', r' ', text)
            sent_tokens = sentence_tokenize(text)  # tokenize the text in terms of sentences
            sent_tokens = [re.sub(r'\s+', ' ', sent) for sent in sent_tokens]

            total_lines += len(sent_tokens)  # count number of sentences before filter
            for sent in sent_tokens:
                total_words += len(word_tokenize(sent))  # count number of words before filter

            sent_tokens = old_mm_filter(
                sent_tokens)  # filter the sentences based on ones that contain "select words"
            print(sent_tokens)
            save_csv(sent_tokens, f_name, output_path)  # store the filtered sentences into csv files

            filtered_lines += len(sent_tokens)  # number of filtered sentences
            for sent in sent_tokens:
                filtered_words += len(word_tokenize(sent))  # count number of words before filter

    return {"Total Lines": total_lines, "Total Words": total_words,
            "Filtered Lines": filtered_lines, "Filtered Words": filtered_words}


def pc_call(input_path, output_path):

    total_lines = 0
    total_words = 0
    filtered_lines = 0
    filtered_words = 0

    for f_name in os.listdir(input_path):
        file_path = os.path.join(input_path, f_name)
        if os.path.isfile(file_path) and not f_name.startswith("."):
            print(f_name)
            file = pd.read_csv(file_path)
            file.drop(file.columns[file.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
            sent_tokens = file['sentence'].tolist()

            total_lines += len(sent_tokens)  # count number of sentences before filter
            for sent in sent_tokens:
                total_words += len(word_tokenize(sent))  # count number of words before filter

            sent_tokens = dictionary_filter(
                sent_tokens)  # filter the sentences based on ones that contain "select words"
            print(sent_tokens)
            save_csv(sent_tokens, f_name, output_path)  # store the filtered sentences into csv files

            filtered_lines += len(sent_tokens)  # number of filtered sentences
            for sent in sent_tokens:
                filtered_words += len(word_tokenize(sent))  # count number of words before filter

    return {"Total Lines": total_lines, "Total Words": total_words,
            "Filtered Lines": filtered_lines, "Filtered Words": filtered_words}


def sp_call(input_path, output_path):
    total_lines = 0
    total_words = 0
    filtered_lines = 0
    filtered_words = 0
    total_files = len(os.listdir(input_path))

    for f_name in os.listdir(input_path):
        file_path = os.path.join(input_path, f_name)
        if os.path.isfile(file_path) and not f_name.startswith("."):
            print(f_name)
            with open(file_path, errors='ignore') as f:
                content = f.readlines()
            sent_tokens = [x.strip() for x in content]

            total_lines += len(sent_tokens)  # count number of sentences before filter
            for sent in sent_tokens:
                total_words += len(word_tokenize(sent))  # count number of words before filter

            sent_tokens = dictionary_filter(
                sent_tokens)  # filter the sentences based on ones that contain "select words"
            save_csv(sent_tokens, f_name, output_path)  # store the filtered sentences into csv files

            filtered_lines += len(sent_tokens)  # number of filtered sentences
            for sent in sent_tokens:
                filtered_words += len(word_tokenize(sent))  # count number of words before filter

    return {"Total Lines": total_lines, "Total Words": total_words,
            "Filtered Lines": filtered_lines, "Filtered Words": filtered_words, "Total Files": total_files}


if __name__ == "__main__":
    '''
    mm = mm_call(input_path="../data/raw_data/meeting_minutes/",
                 output_path="../data/filtered_data/meeting_minutes/")
    '''
    pc = pc_call(input_path="../data/raw_data/press_conference/csv/select/",
                 output_path="../data/filtered_data/press_conference/")

    '''
    sp_all = sp_call(input_path="../data/raw_data/speech/text/all/",
                     output_path="../data/filtered_data/speech/all/")
    sp_non_select = sp_call(input_path="../data/raw_data/speech/text/non-select/",
                            output_path="../data/filtered_data/speech/non-select/")
    sp_select = sp_call(input_path="../data/raw_data/speech/text/select/",
                        output_path="../data/filtered_data/speech/select/")
    '''

#print(mm['Total Lines'])
#print(mm['Total Words'])
#print(mm['Filtered Lines'])
#print(mm['Filtered Words'])
print("#---------------------#")
print(pc['Total Lines'])
print(pc['Total Words'])
print(pc['Filtered Lines'])
print(pc['Filtered Words'])
print("#---------------------#")
#print(sp_all['Total Lines'])
#print(sp_all['Total Words'])
#print(sp_all['Filtered Lines'])
#print(sp_all['Filtered Words'])
#print(sp_all['Total Files'])
print("#---------------------#")
#print(sp_non_select['Total Lines'])
#print(sp_non_select['Total Words'])
#print(sp_non_select['Filtered Lines'])
#print(sp_non_select['Filtered Words'])
#print(sp_non_select['Total Files'])
#print("#---------------------#")
#print(sp_select['Total Lines'])
#print(sp_select['Total Words'])
#print(sp_select['Filtered Lines'])
#print(sp_select['Filtered Words'])
#print(sp_select['Total Files'])
