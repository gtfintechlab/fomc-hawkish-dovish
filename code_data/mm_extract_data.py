import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

file_path = pd.read_excel("../data/master_files/master_meeting_minutes_1996_2020_cleanmeta_NLP_v4.xlsx")['Url'].tolist()


def export_text(text, name):
    file = open("../data/raw_data/meeting_minutes/" + name + ".txt", "w", encoding="utf-8")
    file.write(text)
    file.close()


def get_files(input_path):
    for file in input_path:
        file_name = re.sub("[^0-9]", "", file.split("/")[-1])  # get file name that follows yyyy-mm-dd format
        print(file_name)
        url = "https://www.federalreserve.gov/" + file
        res = requests.get(url)
        soup = BeautifulSoup(res.content, "html.parser")
        soup_text = soup.find_all("p")
        soup_text = soup.get_text()
        soup_text = re.sub(r'(\r\n|\r|\n)', ' ', soup_text)

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

        export_text(text=clean_text, name=file_name)  # save text string into a .txt file


if __name__ == "__main__":
    get_files(input_path=file_path)
