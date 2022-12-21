import pandas as pd
import urllib.request as url
import requests
from bs4 import BeautifulSoup as bs
import re
import os
from time import sleep
from nltk.tokenize import sent_tokenize

A1 = ["inflation expectation", "interest rate", "bank rate", "fund rate", "price", "economic activity", "inflation",
      "employment"]
B1 = ["unemployment", "growth", "exchange rate", "productivity", "deficit", "demand", "job market", "monetary policy"]


def title_check(sentence):
    if any(word1 in sentence.lower() for word1 in A1) or any(word2 in sentence.lower() for word2 in B1):
        return True
    return False

def download_speeches():
    master_df = pd.read_csv('../data/master_files/master_speech.csv')
    column_list = list(master_df.columns)
    column_list.append('LocalPath')
    res_list = []
    for index, row in master_df.iterrows():
        if not index % 10:
            print(index)
        try:

            file_name_short = row['Url'].split('/')
            file_name = '../data/raw_data/speech/html/all/' + file_name_short[-1]
            if file_name_short[-1] == 'default.htm':
                file_name = '../data/raw_data/speech/html/all/' + file_name_short[-2] + '.htm'

            print(file_name)
            if not os.path.isfile(file_name):
                post_fix = row['Url']
                curr_url = 'https://www.federalreserve.gov' + post_fix
                r = requests.get(curr_url)
                with open(file_name, 'wb') as outfile:
                    outfile.write(r.content)
                if title_check(row['Title']):
                    with open(file_name.replace("/all/", "/select/"), 'wb') as outfile:
                        outfile.write(r.content)
                else:
                    with open(file_name.replace("/all/", "/non-select/"), 'wb') as outfile:
                        outfile.write(r.content)
                sleep(3)

            curr_row = list(row)
            curr_row.append(file_name)
            res_list.append(curr_row)

        except Exception as e:
            print(e)
    result_df = pd.DataFrame(res_list, columns=column_list)
    result_df.to_csv('../data/master_files/master_speech.csv', index=False)


def get_speech_text():
    master_df = pd.read_csv('../data/master_files/master_speech.csv')  # .iloc[start:end,:]
    column_list = list(master_df.columns)
    column_list.append('txt_path')
    for index, row in master_df.iterrows():
        if not index % 10:
            print(index)
        try:
            file_name = row['LocalPath']
            file_name_split = file_name.split('/')
            f = open(file_name, 'r', encoding="windows-1252",
                     errors='ignore')  # f = open(file_name, 'r', encoding="windows-1252")
            response = f.read()
            f.close()
            soup = bs(response, 'html.parser')

            # extract text (strip all html)
            # Change later: better to extract each p tag and find section it belongs
            soup_text = soup.get_text()
            soup_text = re.sub(r'(\r\n|\r|\n)', ' ', soup_text)
            soup_text = re.sub(r'&nbsp;', ' ', soup_text)
            soup_text = re.sub(r'&#160;', ' ', soup_text)
            location_of_speech = row['Location']
            location_of_speech = location_of_speech.strip()
            title_of_speech = row['Title']
            title_of_speech = title_of_speech.strip()
            speaker_of_speech = row['Speaker']
            speaker_of_speech = speaker_of_speech.strip()

            # clean text
            # start from Title, or location
            # end with 1. "return to top" 2. "Footnote" 3. "References" 4. "Return to text" 5. "Last Update"
            # some files have early "return to top": e.g., 19970121
            # manually modified:
            found_bool = False
            m = re.search(location_of_speech + '(.*?)return\sto\stop', soup_text, re.IGNORECASE)
            if m:
                found_bool = True
            if found_bool == False:
                m = re.search(location_of_speech + '(.*?)References', soup_text, re.IGNORECASE)
            if m:
                found_bool = True
            if found_bool == False:
                m = re.search(location_of_speech + '(.*?)Footnote', soup_text, re.IGNORECASE)
            if m:
                found_bool = True
            if found_bool == False:
                m = re.search(location_of_speech + '(.*?)Return\sto\stext', soup_text, re.IGNORECASE)
            if m:
                found_bool = True
            if found_bool == False:
                m = re.search(location_of_speech + '(.*?)last\supdate', soup_text, re.IGNORECASE)
            if m:
                found_bool = True

            if found_bool == False:
                m = re.search(title_of_speech + '(.*?)return\sto\stop', soup_text, re.IGNORECASE)
            if m:
                found_bool = True
            if found_bool == False:
                m = re.search(title_of_speech + '(.*?)References', soup_text, re.IGNORECASE)
            if m:
                found_bool = True
            if found_bool == False:
                m = re.search(title_of_speech + '(.*?)Footnote', soup_text, re.IGNORECASE)
            if m:
                found_bool = True
            if found_bool == False:
                m = re.search(title_of_speech + '(.*?)Return\sto\stext', soup_text, re.IGNORECASE)
            if m:
                found_bool = True
            if found_bool == False:
                m = re.search(title_of_speech + '(.*?)last\supdate', soup_text, re.IGNORECASE)
            if m:
                found_bool = True

            if found_bool == False:
                m = re.search(speaker_of_speech + '(.*?)return\sto\stop', soup_text, re.IGNORECASE)
            if m:
                found_bool = True
            if found_bool == False:
                m = re.search(speaker_of_speech + '(.*?)References', soup_text, re.IGNORECASE)
            if m:
                found_bool = True
            if found_bool == False:
                m = re.search(speaker_of_speech + '(.*?)Footnote', soup_text, re.IGNORECASE)
            if m:
                found_bool = True
            if found_bool == False:
                m = re.search(speaker_of_speech + '(.*?)Return\sto\stext', soup_text, re.IGNORECASE)
            if m:
                found_bool = True
            if found_bool == False:
                m = re.search(speaker_of_speech + '(.*?)last\supdate', soup_text, re.IGNORECASE)
            if m:
                found_bool = True

            if found_bool == True:
                clean_text = m.group(1)
            else:
                print("Check", file_name)
                continue

            list_of_sentences = sent_tokenize(clean_text)
            list_of_sentences = [re.sub(r'\s+', ' ', sent) for sent in list_of_sentences]
            ## save txt file
            txt_file_name_split = file_name_split[-1].split('.')
            txt_file_name = '../data/raw_data/speech/text/all/' + txt_file_name_split[0] + '.txt'
            with open(txt_file_name, 'w+', encoding="utf-8") as f:
                for item in list_of_sentences[1:-3]:
                    f.write("%s\n" % item)

            if title_check(title_of_speech):
                with open(txt_file_name.replace("/all/", "/select/"), 'w+', encoding="utf-8") as f:
                    for item in list_of_sentences[1:-3]:
                        f.write("%s\n" % item)
            else:
                with open(txt_file_name.replace("/all/", "/non-select/"), 'w+', encoding="utf-8") as f:
                    for item in list_of_sentences[1:-3]:
                        f.write("%s\n" % item)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    download_speeches()
    get_speech_text()

