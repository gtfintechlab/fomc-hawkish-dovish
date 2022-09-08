import pandas as pd
import os

samples = []
directory = "/Users/suvanpaturi/Documents/Meeting-Minutes-Filtered"
for f_name in os.listdir(directory):
    file_path = os.path.join(directory, f_name)
    if os.path.isfile(file_path) and not f_name.startswith("."):
        df = pd.read_csv('/Users/suvanpaturi/Documents/Meeting-Minutes-Filtered/' + f_name)
        sampled_sentences = df.sample(n=5, random_state=1)['sentence'].tolist()
        print(sampled_sentences)  # from each file select 5 random sentences and
        samples.extend(sampled_sentences)  # appends each sentence to samples list

df = pd.DataFrame(samples, columns=["sentence"])  # convert samples list to df and then to csv
df.to_excel('/Users/suvanpaturi/Documents/Meeting-Minutes-Datasets/manual.xlsx')
