import pandas as pd
import os
import re


def save_excel(samples, type):
    df = pd.DataFrame(
        {"sentence": samples[0], "year": samples[1], "label": samples[2]})  # convert samples list to df and then to csv
    if type == 0:  # Meeting Minutes
        df.to_excel("../data/annotated_data/manual-mm.xlsx")
    if type == 1:  # Press Conferences
        df.to_excel("../data/annotated_data/manual-pc.xlsx")
    if type == 2:  # Speeches
        df.to_excel("../data/annotated_data/manual-sp.xlsx")


def sample(input_path):
    samples = []
    years = []
    label = []
    for f_name in os.listdir(input_path):
        file_path = os.path.join(input_path, f_name)
        if os.path.isfile(file_path) and not f_name.startswith("."):
            df = pd.read_csv(input_path + f_name)
            if len(df) < 5:
                num = len(df['sentence'].tolist())
                samples.extend(df['sentence'].tolist())
                years.extend([int(re.sub("[^0-9]", "", f_name)[:4]) for _ in range(num)])
                label.extend(["-" for _ in range(num)])

            else:
                sampled_sentences = df.sample(n=5, random_state=1)['sentence'].tolist()
                # print(sampled_sentences)  # from each file select 5 random sentences and
                samples.extend(sampled_sentences)  # appends each sentence to samples list
                years.extend([int(re.sub("[^0-9]", "", f_name)[:4]) for _ in range(5)])
                label.extend(["-" for _ in range(5)])
    print(years)
    print(len(years))
    print(len(samples))
    print(len(label))
    return sorted(samples), years, label


if __name__ == "__main__":
    mm_samples = sample(input_path="../data/filtered_data/meeting_minutes/")
    print(mm_samples)

    pc_samples = sample(input_path="../data/filtered_data/press_conference/")
    print(pc_samples)

    sp_samples = sample(input_path="../data/filtered_data/speech/select/")
    print(sp_samples)

    save_excel(mm_samples, 0)
    save_excel(pc_samples, 1)
    save_excel(sp_samples, 2)
