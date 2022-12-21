import pandas as pd
import re

data_directory = "../code_data/Meeting-Minutes-Filtered-Labeled/"

master_file_path = "../data/master_files/master_meeting_minutes_1996_2020_cleanmeta_NLP_v4.xlsx"


def get_new_file_path(file):
    file_name = re.sub("[^0-9]", "", file.split("/")[-1])  # gets file name that follows yyyy-mm-dd format
    return data_directory + "labeled_" + file_name + "_filtered" + ".csv"

def calculate_hawkish_dovish_measure(file_path):
    df_meeting_data = pd.read_csv(file_path, usecols=["sentence", "label"])
    
    count_total_sentences = len(df_meeting_data.index)

    # count hawkish sentences
    df_hawkish = df_meeting_data.loc[df_meeting_data['label'] == "LABEL_1"]
    count_hawkish_sentences = len(df_hawkish.index)

    # count dovish sentences
    df_dovish = df_meeting_data.loc[df_meeting_data['label'] == "LABEL_0"]
    count_dovish_sentences = len(df_dovish.index)

    our_measure = (count_hawkish_sentences - count_dovish_sentences)/count_total_sentences

    return our_measure


df_master = pd.read_excel(master_file_path, usecols=["Year", "Date", "StartDate", "EndDate", "ReleaseDate", "Url"])

df_master["labeled_data_path"] = df_master["Url"].apply(lambda x: get_new_file_path(x))

df_master["our_measure"] = df_master["labeled_data_path"].apply(lambda x: calculate_hawkish_dovish_measure(x))

df_master.to_excel("../data/market_data/aggregate_measure.xlsx", index=False)