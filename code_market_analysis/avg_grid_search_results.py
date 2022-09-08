import pandas as pd
import os

for file in os.listdir("../grid_search_results/"):
    file_path = os.path.join("../grid_search_results/", file)
    print(file_path)
    df_current = pd.read_excel(file_path)
    df_avg = df_current.groupby(by=["Learning Rate", "Batch Size"]).mean()
    df_avg.to_excel(os.path.join("../grid_search_results_avg/", file))