import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

files = pd.read_excel("../master_files/master_mm_final_Oct_2024.xlsx")['Url'].tolist()[214:]


def export_text(text, name):
    file = open("../raw_data/meeting_minutes/" + name + ".txt", "w", encoding="utf-8")
    file.write(text)
    file.close()

def get_files(files):
    for file in files:
        file_name = re.sub("[^0-9]", "", file.split("/")[-1])  # gets file name that follows yyyy-mm-dd format
        print(file_name)
        url = "https://www.federalreserve.gov/" + file
        res = requests.get(url)
        soup = BeautifulSoup(res.content, "html.parser")
        # -------------------------------------------------
        # extract text (strip all html)
        # Change later: better to extract each p tag and find section it belongs
        soup_text = soup.find_all("p")
        soup_text = soup.get_text()
        soup_text = re.sub(r'(\r\n|\r|\n)', ' ', soup_text)
        #print(soup_text)

        # clean text
        # start from "The manager of the System Open Market Account" or "SOMA manager" till "Votes for this action" or "Voting for this action"
        # check 2016 1st meeting file, it contains additional info(Annual Organizational Matters) in the beginning.
        found_bool = False
        m = re.search('the manager of the system open market account(.*?)adjourned', soup_text, re.IGNORECASE)
        if m:
            found_bool = True
        if not found_bool:
            m = re.search('SOMA manager(.*?)adjourned', soup_text, re.IGNORECASE)
        if m:
            found_bool = True
        if not found_bool:
            m = re.search('Balance Sheet Normalization(.*?)adjourned', soup_text, re.IGNORECASE)
        if m:
            found_bool = True
        if not found_bool:
            m = re.search('Committee participants resumed(.*?)adjourned', soup_text, re.IGNORECASE)
        if m:
            found_bool = True
        if not found_bool:  # 20181108 case
            m = re.search('Committee participants resumed(.*?)Voting for this action', soup_text, re.IGNORECASE)
        if m:
            found_bool = True
        if not found_bool:
            m = re.search('Committee participants began(.*?)adjourned', soup_text, re.IGNORECASE)
        if m:
            found_bool = True
        if not found_bool:
            m = re.search('The System Open Market Account (SOMA) manager(.*?)adjourned', soup_text, re.IGNORECASE)
        if m:
            found_bool = True
        if not found_bool:  # 20130619 case
            m = re.search('In light of the changes in the System Open Market Account(.*?)adjourned', soup_text,
                          re.IGNORECASE)
        if m:
            found_bool = True
        if not found_bool:  # 20151028 case
            m = re.search('The staff presented several briefings regarding the concept of an equilibrium(.*?)adjourned',
                          soup_text, re.IGNORECASE)
        if m:
            found_bool = True
        if not found_bool:  # 20160727 case
            m = re.search('The staff provided several briefings that reviewed(.*?)adjourned', soup_text, re.IGNORECASE)
        if m:
            found_bool = True
        if not found_bool:  # 2019-2020 onwards
            m = re.search('Developments in Financial Markets and Open Market Operations(.*?)adjourned', soup_text,
                          re.IGNORECASE)
        if m:
            found_bool = True

        if found_bool:
            clean_text = m.group(1)
        else:
            print("Check", file_name)

        #print(clean_text)
        export_text(text=clean_text, name=file_name)  # saves text string into a .txt file for later use

get_files(files)
