import pandas as pd
from get_sentiment import get_sentiment

target_word_list = ["inflation expectation", "interest rate", "bank rate", "fund rate", "price", "economic activity", "inflation",
      "employment", "unemployment", "growth", "exchange rate", "productivity", "deficit", "demand", "job market", "monetary policy"]

df = pd.read_excel("../training_data_old/manual_v2.xlsx")

senti_nomics_df = get_sentiment(text = list(df["sentence"]), include = target_word_list)

#senti_nomics_df["Doc_id"] = senti_nomics_df["Doc_id"] - 1

senti_nomics_df.to_excel("senti_nomics_output.xlsx")
print(senti_nomics_df.head())

df_merge = pd.merge(df, senti_nomics_df, left_on="sentence_index", right_on="Doc_id", how="right")

df_merge.to_excel("manual_v2_split_sentence.xlsx", index=False)

print(df_merge.head())