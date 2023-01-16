import pandas as pd

df = pd.read_excel("../labeled_data/lab-manual-split-combine.xlsx", index_col=False)
df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
df = df[['sentence', 'year', 'label']]
df = df.sort_values(by='year', ascending=True)
train = df[df['year'] < 2020]
test = df[df['year'] >= 2020]

if __name__ == "__main__":
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df), print(train), print(test)
        train.to_excel("../look_ahead/1996-2019-train.xlsx", index=False)
        test.to_excel("../look_ahead/2020-2022-test.xlsx", index=False)