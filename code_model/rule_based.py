import collections

import numpy as np
import pandas as pd
import os
import sklearn.metrics as skm
import re
import numpy as np
from string import digits

## Yuriy Paper
A1 = ["inflation expectation", "interest rate", "bank rate", "fund rate", "price", "economic activity", "inflation",
      "employment"]
A2 = ["anchor", "cut", "subdue", "decline", "decrease", "reduce", "low", "drop", "fall", "fell", "decelarate", "slow",
      "pause", "pausing",
      "stable", "non-accelerating", "downward", "tighten"]

B1 = ["unemployment", "growth", "exchange rate", "productivity", "deficit", "demand", "job market", "monetary policy"]
B2 = ["ease", "easing", "rise", "rising", "increase", "expand", "improve", "strong", "upward", "raise", "high", "rapid"]

C = ["weren't", "were not", "wasn't", "was not", 'did not', "didn't", "do not", "don't", 'will not', "won't"]

dir = "../training_data/test-and-training/test_data/"
output_dir = "../rule_based_results/"
test_dir = sorted(os.listdir(dir))


def rule_model(df):
    sentences = df['sentence'].tolist()
    pred = []
    for s in sentences:
        label = 0
        if (any(word in s.lower() for word in A1) and any(word in s.lower() for word in A2)) or \
                (any(word in s.lower() for word in B1) and any(word in s.lower() for word in B2)):
            label = 0
        elif (any(word in s.lower() for word in A1) and any(word in s.lower() for word in B2)) or \
                (any(word in s.lower() for word in B1) and any(word in s.lower() for word in A2)):
            label = 1
        else:
            label = 2
        if label != 2 and (any(word in s.lower() for word in C)):
            pred.append(1 - label)  # turn 0 to 1, and 1 to 0
        else:
            pred.append(label)

    return pred


remove_digits = str.maketrans('', '', digits)

score_dict = {}
for i, f in enumerate(test_dir):
    file = pd.read_excel(os.path.join(dir, f))

    predicted = rule_model(file)  # make predictions
    actual = [int(x) for x in file['label'].tolist()]  # actual labels
    file['pred_label'] = predicted  # add column for predicted labels
    file = file[['sentence', 'year', 'label', 'pred_label']]
    name = f.replace(".xlsx", "").replace("-test", "")
    seed = int(re.findall("\d+", name)[0])  # find specific seed of file
    base_name = name.translate(remove_digits)[:-1]  # file type (mm, pres conf, speed)
    file.to_excel(output_dir + base_name + "-results-" + str(seed) + ".xlsx", index=False)  # save test results

    cp = skm.classification_report(y_true=actual, y_pred=predicted, output_dict=True)
    print(base_name)
    print(cp['weighted avg']['f1-score'])
    if base_name in score_dict:
        score_dict[base_name].append(cp['weighted avg']['f1-score'])
    else:
        score_dict[base_name] = [cp['weighted avg']['f1-score']]

for data in score_dict:
    score_dict[data] = (np.mean(score_dict[data]), np.std(score_dict[data]))
print(score_dict)

