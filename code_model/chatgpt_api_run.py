import openai,os,sys
openai.api_key = ""
# messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
# ]
import os

import pandas as pd
import numpy as np
from time import sleep

for seed in [5768, 78516, 944601]:
    for data_category in ["lab-manual-combine", "lab-manual-sp", "lab-manual-mm", "lab-manual-pc", "lab-manual-mm-split", "lab-manual-pc-split", "lab-manual-sp-split", "lab-manual-split-combine"]:

        # load training data
        test_data_path = "../training_data/test-and-training/test_data/" + data_category + "-test" + "-" + str(seed) + ".xlsx"
        data_df = pd.read_excel(test_data_path)


        sentences = data_df['sentence'].to_list()
        labels = data_df['label'].to_numpy()

        # exit(0)
        output_list = []
        for i in range(len(sentences)): 
            sen = sentences[i]
            message = "Discard all the previous instructions. Behave like you are an expert sentence classifier. Classify the following sentence from FOMC into 'HAWKISH', 'DOVISH', or 'NEUTRAL' class. Label 'HAWKISH' if it is corresponding to tightening of the monetary policy, 'DOVISH' if it is corresponding to easing of the monetary policy, or 'NEUTRAL' if the stance is neutral. Provide the label in the first line and provide a short explanation in the second line. The sentence: " + sen
            # messages.append(
            #         {"role": "user", "content": message},
            # )
            messages = [
                    {"role": "user", "content": message},
            ]
            try:
                chat_completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=1000
                )
            except Exception as e:
                print(e)
                i = i - 1
                sleep(10.0)

            answer = chat_completion.choices[0].message.content
            
            output_list.append([labels[i], sen, answer])
            sleep(1.0) 

            results = pd.DataFrame(output_list, columns=["true_label", "original_sent", "text_output"])

            results.to_csv(f'../llm_prompt_test_labels/chatgpt_{data_category}_{seed}.csv', index=False)
