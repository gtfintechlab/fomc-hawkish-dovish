from xml.dom.minidom import Document
import pandas as pd
import re


def get_new_file_path(file, document_type):
    file_split = file.split("/")
    file_split_last = file_split[-1]
    file_name = file_split_last.split(".")[0]
    if file_name == "default" and document_type=="speech":
        return data_directory + "labeled_Speech_" + file_split[-2] + "_filtered" + ".csv"
    elif (file_name == "default" or file_name == "testimony") and document_type=="testimony":
        return data_directory + "labeled_Testimony_" + file_split[-3] + "_" + file_split[-2] + "_filtered" + ".csv"
    else:
        return data_directory + "labeled_" + file_name + "_filtered" + ".csv"

def calculate_hawkish_dovish_measure(file_path):
    df_meeting_data = pd.read_csv(file_path)

    count_total_sentences = len(df_meeting_data.index)
    if count_total_sentences > 0:
        df_meeting_data = df_meeting_data[["sentence", "label"]]
        
        # count hawkish sentences
        df_hawkish = df_meeting_data.loc[df_meeting_data['label'] == "LABEL_1"]
        count_hawkish_sentences = len(df_hawkish.index)

        # count dovish sentences
        df_dovish = df_meeting_data.loc[df_meeting_data['label'] == "LABEL_0"]
        count_dovish_sentences = len(df_dovish.index)

        our_measure = (count_hawkish_sentences - count_dovish_sentences)/count_total_sentences

        return our_measure
    else:
        return 0

def calculate_filtered_sent(file_path):
    df_meeting_data = pd.read_csv(file_path)
    
    count_total_sentences = len(df_meeting_data.index)

    return count_total_sentences



document_type = "testimony"#"speech"


data_directory = "../code_data/" + document_type + "-filtered-labeled/"
master_file_path = "../code_data/master_" + document_type + "_1996_2020_NLP_v4.xlsx"



df_master = pd.read_excel(master_file_path, usecols=["Date", "Title", "Speaker", "Location", "Url"])

df_master["labeled_data_path"] = df_master["Url"].apply(lambda x: get_new_file_path(x, document_type))

df_master["our_measure"] = df_master["labeled_data_path"].apply(lambda x: calculate_hawkish_dovish_measure(x))
df_master["number_of_filtered_sent"] = df_master["labeled_data_path"].apply(lambda x: calculate_filtered_sent(x))

df_master.to_excel("../data/market_data/aggregate_measure_" + document_type + ".xlsx", index=False)