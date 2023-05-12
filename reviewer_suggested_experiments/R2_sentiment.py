import numpy as np
import pandas as pd
import os

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import f1_score

# LABEL_2 is positive (dovish), LABEL_1 is neutral, LABEL_0 is negative (hawkish)
tokenizer = AutoTokenizer.from_pretrained("ipuneetrathore/bert-base-cased-finetuned-finBERT", truncation=True)
model = AutoModelForSequenceClassification.from_pretrained("ipuneetrathore/bert-base-cased-finetuned-finBERT")
classifier = pipeline('sentiment-analysis', model= model, tokenizer=tokenizer, device=0, framework="pt") 

results = []

for data_category in ["lab-manual-combine", "lab-manual-sp", "lab-manual-mm", "lab-manual-pc", "lab-manual-mm-split", "lab-manual-pc-split", "lab-manual-sp-split", "lab-manual-split-combine"]:
    f1_list = []
    for seed in [5768, 78516, 944601]:
        test_data_path = "../training_data/test-and-training/test_data/" + data_category + "-test" + "-" + str(seed) + ".xlsx"



        data_df_test = pd.read_excel(test_data_path)
        sentences_test = data_df_test['sentence'].to_list()
        labels_test = data_df_test['label'].to_list()


        output = classifier(sentences_test)

        pred_list = []
        for sent in output:
            if sent['label'] == 'LABEL_2':
                pred_list.append(0)
            elif sent['label'] == 'LABEL_0':
                pred_list.append(1)
            else:
                pred_list.append(2)

        f1_list.append(f1_score(labels_test, pred_list, average='weighted'))
    print(data_category)
    print("f1 score mean: ", format(np.mean(f1_list), '.4f'))
    results.append([data_category, np.mean(f1_list), np.std(f1_list)])

results_df = pd.DataFrame(results, columns=['data_category', 'f1 score mean', 'f1 score std dev'])
results_df.to_csv("finbert_zero_shot_R2.csv", index=False)