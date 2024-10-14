import pandas as pd
import os

files = os.listdir('../grid_search_results')

files_xls = [f for f in files if 'split-combine_roberta-large.xlsx' in f]

for file in files_xls:
    df = pd.read_excel('../grid_search_results/' + file)
    df_temp = df.groupby(['Learning Rate', 'Batch Size'], as_index=False).agg(
    {
        "Val F1 Score": ["mean"],
        "Test F1 Score": ["mean", "std"],
        "Test Accuracy": ["mean"]
    }
    )
    df_temp.columns = ['Learning Rate', 'Batch Size', 'mean Val F1 Score', 'mean Test F1 Score', 'std Test F1 Score', 'mean Test Accuracy']
    # print(df_temp)
    max_element = df_temp.iloc[df_temp['mean Val F1 Score'].idxmax()] 
    print(file)
    print(max_element)
    print(max_element['mean Test F1 Score'])
    print(format(max_element['std Test F1 Score'], '.4f'), "\n")