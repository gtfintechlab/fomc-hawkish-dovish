import pandas as pd
import os
import multiprocessing
import sys
from time import time
from time import sleep
import random

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

train_test_data_path= "../training_data_old/manual_v2.xlsx"

data_df = pd.read_excel(train_test_data_path)
list_of_sentences = data_df['sentence'].to_list()

tokenizer = AutoTokenizer.from_pretrained("../model_data/final_model", do_lower_case=True, do_basic_tokenize=True)

model = AutoModelForSequenceClassification.from_pretrained("../model_data/final_model", num_labels=3)

config = AutoConfig.from_pretrained("../model_data/final_model")

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, config=config, device=0, framework="pt")
results = classifier(list_of_sentences, batch_size=128, truncation="only_first", device=0)

result_df = pd.DataFrame.from_dict(results)

# store into the sub-dataframe
data_df['label_roberta_base'] = result_df['label']
data_df['score_roberta_base'] = result_df['score']

data_df.to_excel("diagnose_label.xlsx", index=False)