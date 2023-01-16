import pandas as pd
import os

files = os.listdir('../grid_search_results')

files_xls = [f for f in files if 'roberta-large.xlsx' in f]

for file in files_xls:
    df = pd.read_excel('../grid_search_results/' + file)
    df_temp = df.groupby(['Learning Rate', 'Batch Size'], as_index=False).mean()
    max_element = df_temp.iloc[df_temp['Val F1 Score'].idxmax()] #df_temp['Val F1 Score'].max()
    print(file)
    print(max_element)
    print(max_element['Test F1 Score'], "\n")