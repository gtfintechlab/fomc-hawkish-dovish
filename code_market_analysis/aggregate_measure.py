import pandas as pd
import re
import numpy as np

data_directory = "../data/filtered_data/press_conference_labeled/"#"../data/filtered_data/speech_labeled/" #"../data/filtered_data/meeting_minutes_labeled/"


def get_new_file_path_mm(file):
    file_name = re.sub("[^0-9]", "", file.split("/")[-1])  # gets file name that follows yyyy-mm-dd format
    return data_directory + "labeled_" + file_name + "_filtered" + ".csv"

def get_new_file_path_sp(file):
    file_split = file.split("/")[-1]
    file_name = file_split.split(".")[0]
    return data_directory + "labeled_" + file_name + "_filtered" + ".csv"

def get_new_file_path_pc(file):
    file_split = file.split("/")[-1]
    file_name = file_split.split(".")[0]
    return data_directory + "labeled_" + file_name + "_select_filtered" + ".csv"

def calculate_hawkish_dovish_measure(file_path):
    try:
        print(file_path)
        df_meeting_data = pd.read_csv(file_path, usecols=["sentence", "label"])
    except:
        return np.nan
    count_total_sentences = len(df_meeting_data.index)
    if count_total_sentences > 0:
        # count hawkish sentences
        df_hawkish = df_meeting_data.loc[df_meeting_data['label'] == "LABEL_1"]
        count_hawkish_sentences = len(df_hawkish.index)

        # count dovish sentences
        df_dovish = df_meeting_data.loc[df_meeting_data['label'] == "LABEL_0"]
        count_dovish_sentences = len(df_dovish.index)

        our_measure = (count_hawkish_sentences - count_dovish_sentences)/count_total_sentences
    else:
        our_measure = 0
    return our_measure

'''
## for meeting minutes
master_file_path = "../data/master_files/master_mm_final.xlsx"

df_master = pd.read_excel(master_file_path, usecols=["Year", "Date", "StartDate", "EndDate", "ReleaseDate", "Url"])

df_master["labeled_data_path"] = df_master["Url"].apply(lambda x: get_new_file_path_mm(x))

df_master["our_measure"] = df_master["labeled_data_path"].apply(lambda x: calculate_hawkish_dovish_measure(x))

df_master.to_excel("../data/market_analysis_data/aggregate_measure_mm.xlsx", index=False)


## for Speeches
master_file_path = "../data/master_files/master_speech_final.csv"

df_master = pd.read_csv(master_file_path, usecols=["Date", "Title", "Speaker", "Location", "LocalPath"])

df_master["labeled_data_path"] = df_master["LocalPath"].apply(lambda x: get_new_file_path_sp(x))

print(df_master.shape)

df_master["our_measure"] = df_master["labeled_data_path"].apply(lambda x: calculate_hawkish_dovish_measure(x))

df_master = df_master.dropna()
df_master = df_master.drop_duplicates("labeled_data_path")
print(df_master.shape)

df_master.to_excel("../data/market_analysis_data/aggregate_measure_sp.xlsx", index=False)
'''


## for PC
master_file_path = "../data/master_files/master_pc_final.xlsx"

df_master = pd.read_excel(master_file_path, usecols=["Year", "Date", "StartDate", "EndDate", "TranscriptUrl"])

df_master["labeled_data_path"] = df_master["TranscriptUrl"].apply(lambda x: get_new_file_path_pc(x))

df_master["our_measure"] = df_master["labeled_data_path"].apply(lambda x: calculate_hawkish_dovish_measure(x))

print(df_master.shape)

df_master.to_excel("../data/market_analysis_data/aggregate_measure_pc.xlsx", index=False)
