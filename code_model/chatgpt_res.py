import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def decode(x):
    if x[:6] == "DOVISH":
        return 0
    elif x[:7] == "HAWKISH":
        return 1
    else: 
        return 2

for data_category in ["lab-manual-combine", "lab-manual-sp", "lab-manual-mm", "lab-manual-pc", "lab-manual-mm-split", "lab-manual-pc-split", "lab-manual-sp-split", "lab-manual-split-combine"]:
    acc_list = []
    f1_list = []
    for seed in [5768, 78516, 944601]:
        df = pd.read_csv(f'../llm_prompt_test_labels/chatgpt_{data_category}_{seed}.csv')
    
        df["predicted_label"] = df["text_output"].apply(lambda x: decode(x))
        acc_list.append(accuracy_score(df["true_label"], df["predicted_label"]))
        f1_list.append(f1_score(df["true_label"], df["predicted_label"], average='weighted'))

    print(data_category)
    print("f1 score mean: ", format(np.mean(f1_list), '.4f'))
    print("f1 score std: ", format(np.std(f1_list), '.4f'), "\n")

    