import re
import pandas as pd

A1 = ["inflation expectation", "interest rate", "bank rate", "fund rate", "price", "economic activity", "inflation",
      "employment"]
B1 = ["unemployment", "growth", "exchange rate", "productivity", "deficit", "demand", "job market", "monetary policy"]


def dict_check(sent):
    word_check = True
    for s in sent:
        if any(a in s.lower() for a in A1) or any(b in s.lower() for b in B1):
            word_check = word_check and True
        else:
            word_check = word_check and False

    return word_check


def create_excel(df, output_path):
    df.to_excel(output_path)


def sentence_split_df(input_path):
    df = pd.read_excel(input_path)
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    print(df)

    sentences = df['sentence'].tolist()
    orig_index = df.index.values.tolist()
    curr_labels = df['label'].tolist()
    years = df['year'].tolist()
    split_indexes = []

    print(sentences)
    print(orig_index)
    print(curr_labels)

    num_times = 0
    for index, row in df.iterrows():
        i = int(index)
        sentence = row['sentence']
        year = row['year']
        temp_split = re.split(r' but | however | even though | although | while |;', sentence)
        temp_split = [w.strip() for w in temp_split]
        if "," in temp_split:
            temp_split.remove(",")
        if "" in temp_split:
            temp_split.remove("")

        if dict_check(temp_split) and len(temp_split) > 1:
            num_times += 1
            split_indexes.append(i)
            sentences.extend(temp_split)
            orig_index.extend([i] * len(temp_split))
            curr_labels.extend(["-"] * len(temp_split))
            years.extend([year] * len(temp_split))

    print(len(sentences))
    print(len(years))
    print(len(curr_labels))
    print(len(orig_index))

    df2 = pd.DataFrame({'sentence': sentences,
                        'year': years,
                        'label': curr_labels,
                        'orig_index': orig_index
                        })

    print(split_indexes)
    df2 = df2.loc[~((df2['orig_index'].isin(split_indexes)) & (df2['label'] != '-')), :]
    df2 = df2.sort_values(by=['orig_index'], ascending=True)
    df2 = df2.reset_index(drop=True)
    print(df2)
    print(num_times)
    print(len(df2))
    return df2


if __name__ == "__main__":
    mm_df = sentence_split_df("../labeled_data/lab-manual-mm.xlsx")
    create_excel(mm_df, "../data/annotated_data/manual-mm-split.xlsx")

    pc_df = sentence_split_df("../labeled_data/lab-manual-pc.xlsx")
    create_excel(pc_df, "../data/annotated_data/manual-pc-split.xlsx")

    sp_df = sentence_split_df("../labeled_data/lab-manual-sp.xlsx")
    create_excel(sp_df, "../data/annotated_data/manual-sp-split.xlsx")

