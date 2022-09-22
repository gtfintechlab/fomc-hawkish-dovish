import pandas as pd
import os

samples = []

directory = "./Meeting-Minutes-Filtered"
for f_name in os.listdir(directory):
    file_path = os.path.join(directory, f_name)
    if os.path.isfile(file_path) and not f_name.startswith("."):
        df = pd.read_csv('./Meeting-Minutes-Filtered/' + f_name)
        sampled_sentences = df.sample(n=5, random_state=1)['sentence'].tolist()
        #print(sampled_sentences)  # from each file select 5 random sentences and
        samples.extend(sampled_sentences)  # appends each sentence to samples list

df = pd.DataFrame(samples, columns=["sentence"])  # convert samples list to df and then to csv

df.to_excel('../../training_data/manual_new.xlsx')


df = df.drop_duplicates("sentence")
print(df.shape)
#### Verify generated sample 
df_annotated = pd.read_excel("../../training_data/manual_v2.xlsx")
df_annotated = df_annotated.drop_duplicates("sentence")
print(df_annotated.shape)

df_merged = pd.merge(df, df_annotated, on="sentence", how="inner")
print(df_merged.shape)