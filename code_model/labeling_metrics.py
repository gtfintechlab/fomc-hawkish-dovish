import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from collections import Counter

pd.set_option("display.max_rows", None, "display.max_columns", None)

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


def negation_check(sentence):
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


df = pd.read_excel("/Users/suvanpaturi/Documents/Meeting-Minutes-Datasets/manual.xlsx")
sentences = df['sentence'].tolist()
manual_labels = df['label'].tolist()
print(sentences)
print(manual_labels)
predicted_labels = []
for sentence in sentences:
    predicted_labels.append(labeling(sentence))

print(Counter(manual_labels))
print(Counter(predicted_labels))
print(classification_report(manual_labels, predicted_labels))
print(accuracy_score(manual_labels, predicted_labels))
df['predicted_label'] = predicted_labels
df.to_excel('/Users/suvanpaturi/Documents/Meeting-Minutes-Datasets/temp.xlsx')
