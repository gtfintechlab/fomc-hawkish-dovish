import os
import re
import urllib.request
from io import StringIO
from time import sleep

import pandas as pd
from nltk.tokenize import sent_tokenize
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from urllib3.util import url

input_file_path = "../data/raw_data/press_conference/pdf/"
output_file_path = "../data/raw_data/press_conference/csv/"


def download_meeting_press_conference():
    master_df = pd.read_excel(
        '../data/master_files/master_meeting_press_conference_transcripts_2011_2020_cleanmeta.xlsx')  # .iloc[start:end,:]
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'whatever')
    for index, row in master_df.iterrows():
        if not index % 10:
            print(index)
        try:
            file_name_short = row['TranscriptUrl'].split('/')
            file_name = '../data/raw_data/press_conference/pdf/' + file_name_short[-1]

            print(file_name)
            if not os.path.isfile(file_name):
                post_fix = row['TranscriptUrl']
                curr_url = 'https://www.federalreserve.gov' + post_fix
                opener.retrieve(curr_url, file_name)
                sleep(3)

        except Exception as e:
            print(e)


def convert_pdf_to_string(file_path):
    output_string = StringIO()
    with open(file_path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    return output_string.getvalue()


def convert_title_to_filename(title):
    filename = title.lower()
    filename = filename.replace(' ', '_')
    return filename


def split_to_title_and_page_num(table_of_contents_entry):
    title_and_page_num = table_of_contents_entry.strip()

    title = None
    page_num = None

    if len(title_and_page_num) > 0:
        if title_and_page_num[-1].isdigit():
            i = -2
            while title_and_page_num[i].isdigit():
                i -= 1

            title = title_and_page_num[:i].strip()
            page_num = int(title_and_page_num[i:].strip())

    return title, page_num


def sentence_tokenize(text):
    return sent_tokenize(text.replace("\n", ""))


def save_csv(df, name, location):
    new_name = name.replace(".pdf", "")
    df.to_csv(location + new_name + ".csv")


def get_all_files(input_path, output_path):
    lst = os.listdir(input_path)
    lst.sort()

    for f_name in lst:
        file_path = os.path.join(input_path, f_name)
        if os.path.isfile(file_path) and not f_name.startswith("."):
            print(file_path)
            text = convert_pdf_to_string(file_path)
            max_pages = split_to_title_and_page_num(text)[1]  # max number of pages

            # ------------------------- initial cleaning and tokenization process
            t = re.sub(r'(\r\n|\r|\n)', ' ', text)
            t = re.sub('  ', ' ', t)
            t = re.sub(r'(?<=[.,;])(?=[^\s])', r' ', t)
            sent_tokens = sentence_tokenize(text)  # tokenize the text in terms of sentences
            sent_tokens = [re.sub(r'\s+', ' ', sent) for sent in sent_tokens]
            print(sent_tokens)

            # ------------------------ cleaning process by removing title and page text in sentences
            temp = sent_tokens[0].split(" ")

            # Ex.
            # May 4, 2022 Chair Powell’s Press Conference PRELIMINARY
            # Transcript of Chair Powell’s Press Conference May 4, 2022 CHAIR POWELL.

            t = temp.index("Transcript")
            press_title = ' '.join(temp[:t])  # May 4, 2022 Chair Powell’s Press Conference PRELIMINARY
            print(press_title)

            speaker = temp[-2] + " " + (temp[-1])[:-1]  # CHAIR POWELL

            speakers = []
            sentences = []
            print(sent_tokens)
            for i in range(1, len(sent_tokens)):
                s = sent_tokens[i]
                if press_title in s:
                    s = re.sub(press_title, '', s)
                sub = r'Page.+\b{}\b'.format(str(max_pages))
                s = re.sub(sub, '', s)
                for j in range(1, max_pages + 1):
                    sub = r'\b{}\b.+\b{}\b'.format(str(j), str(max_pages))
                    s = re.sub(sub, '', s)
                    s = s.strip().replace("  ", " ").replace("\n", "")
                if s.strip().replace(" ", "")[:-1].isupper():  # CHAIR YELLEN. -> CHAIRYELLEN check if uppercase
                    speaker = s.strip().replace("\n", "")[:-1]  # set the speaker
                    continue
                # -----------------------------
                print(s)
                print(speaker)
                speakers.append(speaker)
                sentences.append(s)
            file = pd.DataFrame({'speaker': speakers, 'sentence': sentences})
            file.drop(file.tail(1).index, inplace=True)
            select_file = file[file['speaker'].str.contains("CHAIR") & (~file['sentence'].str.contains("\?"))]

            save_csv(df=file, name=f_name, location=output_path + "/all/")
            save_csv(df=select_file, name=f_name + "_select", location=output_path + "/select/")


if __name__ == "__main__":
    get_all_files(input_path=input_file_path, output_path=output_file_path)
