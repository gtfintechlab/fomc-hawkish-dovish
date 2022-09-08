import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import torch
from torch.utils.data import TensorDataset, DataLoader

SEED=5768

torch.manual_seed(SEED)
np.random.seed(SEED) 

df = pd.read_excel("diagnose_label.xlsx")

y_test_full = df["label"].to_list()
y_pred_full = df["label_roberta_base"].to_list()

matrix = confusion_matrix(y_test_full, y_pred_full)
print(matrix)

target_names = ["Dovish", "Hawkish", "Neutral"]
print(classification_report(y_test_full, y_pred_full, target_names=target_names, digits=4))

dataset = TensorDataset(torch.LongTensor(np.array(y_test_full)), torch.LongTensor(np.array(y_pred_full)))

train, val, test = torch.utils.data.random_split(dataset=dataset, lengths=[700, 100, 200])
dataloaders_dict = {'train': DataLoader(train), 
                    'val': DataLoader(val),
                    'test': DataLoader(test)}
y_test = []
y_pred = []
for label, label_roberta_base in dataloaders_dict['test']:
    y_test.append(label.tolist()[0])
    y_pred.append(label_roberta_base.tolist()[0])


print(classification_report(y_test, y_pred, target_names=target_names, digits=4))