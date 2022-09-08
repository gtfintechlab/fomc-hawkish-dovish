import pandas as pd


def file_format(file):
    f = pd.read_excel(file)
    f.drop(f.columns[f.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    return f


def combine_excel(file1, file2, output_name):
    df = pd.merge(file1, file2, how="outer", on=["sentence", "label"])
    df.to_excel(output_name)


mm_file = file_format("/Users/suvanpaturi/Documents/Label-Data/manual_v2.xlsx")
pc_file = file_format("/Users/suvanpaturi/Documents/Label-Data/manual-pc.xlsx")


mm_split_file = file_format("/Users/suvanpaturi/Documents/Label-Data/manual_mm_split.xlsx")
pc_split_file = file_format("/Users/suvanpaturi/Documents/Label-Data/manual-pc-split.xlsx")

combine_excel(mm_file, pc_file, "/Users/suvanpaturi/Documents/Label-Data/manual-combined.xlsx")
combine_excel(mm_split_file, pc_split_file, "/Users/suvanpaturi/Documents/Label-Data/manual-split-combined.xlsx")