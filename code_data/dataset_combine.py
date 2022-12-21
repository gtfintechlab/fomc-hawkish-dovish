import pandas as pd


def file_format(file):
    f = pd.read_excel(file)
    f.drop(f.columns[f.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    return f


mm = file_format("../labeled_data/lab-manual-mm.xlsx")
pc = file_format("../labeled_data/lab-manual-pc.xlsx")
sp = file_format("../labeled_data/lab-manual-sp.xlsx")

mm_split = file_format("../labeled_data/lab-manual-mm-split.xlsx")
pc_split = file_format("../labeled_data/lab-manual-pc-split.xlsx")
sp_split = file_format("../labeled_data/lab-manual-sp-split.xlsx")


def combine(mm, pc, sp):
    temp = pd.concat([mm, pc, sp], axis=0).sort_values(by=['sentence']).reset_index()
    return temp[temp.columns[~temp.columns.isin(['assists', 'rebounds'])]]



if __name__ == "__main__":
    manual_combine = combine(mm=mm, pc=pc, sp=sp)
    manual_combine.to_excel("../labeled_data/lab-manual-combine.xlsx")
    split_combine = combine(mm=mm_split, pc=pc_split, sp=sp_split)
    split_combine.to_excel("../labeled_data/lab-manual-split-combine.xlsx")
