import pandas as pd
import os
import sklearn.model_selection as sk

seeds = [5768, 78516, 944601]


def seed_split(input_path, seed):
    lst = os.listdir(input_path)
    for f_name in lst:
        if not f_name.startswith("."):
            file_path = os.path.join(input_path, f_name)
            file_name = f_name.replace(".xlsx", "")
            file = pd.read_excel(file_path)
            edge_values = list(
                filter(lambda a: a not in [0, 1, 2], file['label'].tolist()))  # use to see if invalid labels exist
            if len(edge_values) > 0:
                print(file_name)
            train, test = sk.train_test_split(file, train_size=0.8, random_state=seed)
            train.drop(train.columns[train.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
            test.drop(test.columns[test.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
            train.to_excel("../training_data/" + file_name + "-train-" + str(seed) + ".xlsx", index=False)
            test.to_excel("../test_data/" + file_name + "-test-" + str(seed) + ".xlsx", index=False)


for seed in seeds:
    seed_split("../labeled_data/", seed)
