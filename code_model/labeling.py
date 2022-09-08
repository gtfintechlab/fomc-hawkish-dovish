import pandas as pd
import os

directory = "/Users/suvanpaturi/Documents/Meeting-Minutes-Filtered"
frames = []
for f_name in os.listdir(directory):
    file_path = os.path.join(directory, f_name)
    if os.path.isfile(file_path) and not f_name.startswith("."):
        df = pd.read_csv('/Users/suvanpaturi/Documents/Meeting-Minutes-Filtered/' + f_name)
        frames.append(df)


def merge_data(frames):
    result = pd.concat(frames, ignore_index=True, sort=False).reset_index()
    return result


# -----------------BASE LABELING------------------

A1 = ["inflation expectation", "interest rate", "bank rate", "fund rate", "price", "economic activity",
      "inflation",
      "employment"]
A2 = ["anchor", "cut", "subdue", "decline", "decrease", "reduce", "low", "drop", "fall", "fell",
      "decelerate", "slow",
      "pause", "pausing",
      "stable", "non-accelerating", "downward", "tighten"]
B1 = ["unemployment", "growth", "exchange rate", " productivity", " deficit", " demand", " job market",
      "monetary policy"]
B2 = ["ease", "easing", "rise", "rising", "increase", "expand", "improve", "strong", "upward", "raise",
      "high", "rapid"]
negation = ["weren't", "were not", "wasn't", "was not", "did not", "didn't", "do not", "don't", "will not",
            "won't"]

# -----------------DEFINED LABELING------------------------------
defined_pharses = ["target", "risk", "federal funds rate", "CPI", "dollar", "bonds", "spreads", "grow",
                   "monetary policy"]
dovish_phrases = ["accommodative", "accomodation", "lower", "downside", "stimulus",
                  "resource slack", "underutilize", "depreciate", "negative effect", "narrow", "slowed"]
neutral_phrases = ["unchanged", "maintain", "leave", "maintain", "keep", "stable",
                   "balanced", "mixed"]
hawkish_phrases = ["raise" "upside", "oil", "energy"
                                            "return", "appreciate", "adjustments", "widen", "accelerate"]


def labeling(sentence):
    # BASE LABELING

    count_zero, count_one, count_neutral = 0, 0, 0

    if any(word1 in sentence for word1 in A1) and any(word2 in sentence for word2 in A2):
        if negation_check(sentence):
            count_one += 1
        else:
            count_zero += 1
    if any(word1 in sentence for word1 in A2) and any(word2 in sentence for word2 in B1):
        if negation_check(sentence):
            count_zero += 1
        else:
            count_one += 1
    if any(word1 in sentence for word1 in A1) and any(word2 in sentence for word2 in B2):
        if negation_check(sentence):
            count_zero += 1
        else:
            count_one += 1
    if any(word1 in sentence for word1 in B1) and any(word2 in sentence for word2 in B2):
        if negation_check(sentence):
            count_one += 1
        else:
            count_zero += 1

    # DEFINITE LABELING (More weightage)
    if any(word in sentence for word in dovish_phrases) and any(word in sentence for word in defined_pharses):
        if negation_check(sentence):
            count_one += 2
        else:
            count_zero += 2
    if any(word in sentence for word in neutral_phrases):
        count_neutral += 2
    if any(word in sentence for word in hawkish_phrases) and any(word in sentence for word in defined_pharses):
        if negation_check(sentence):
            count_zero += 2
        else:
            count_one += 2

    return aggregation_function(labels=[count_zero, count_one, count_neutral])


def negation_check(sentence):  # how to account for double negations
    return any(word in sentence for word in negation)


def aggregation_function(labels):
    count_zero = labels[0]
    count_one = labels[1]
    count_neutral = labels[2]
    if (max(count_zero, count_neutral, count_one) == count_neutral) or (count_zero == count_one):
        return 2
    elif max(count_zero, count_neutral, count_one) == count_zero:
        return 0
    elif max(count_zero, count_neutral, count_one) == count_one:
        return 1


def apply_label(df):
    labels = []
    for index, row in df.iterrows():
        labels.append(labeling(row['sentence']))
    df['label'] = labels
    return df


df = merge_data(frames)
labeled_df = apply_label(df)
labeled_df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
labeled_df.drop(df.columns[df.columns.str.contains('index', case=False)], axis=1, inplace=True)
print(labeled_df)
labeled_df.to_excel('/Users/suvanpaturi/Documents/Meeting-Minutes-Datasets/labeled.xlsx')
