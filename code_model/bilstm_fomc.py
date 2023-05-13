"""
Author: Suvan Paturi
"""

import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import sklearn.model_selection as sk
import sklearn.metrics as skm

# Text pre-processing
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout, GlobalAveragePooling1D, Flatten, \
    SpatialDropout1D, Bidirectional
import string
from string import digits
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str("0")

# -----------------------------------------------------------

output_dir = "../lstm_results/"
pd.set_option('display.max_rows', 500)




def get_max_length(df):
    max = 0
    for index, row in df.iterrows():  # format sentence for tokenization
        sentence = row['sentence'].replace(",", "").replace(".", " ") \
            .replace("—", " ").replace("â€", "").replace("  ", " ") \
            .replace(";", "").replace("\n", " ").translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(sentence)
        if len(words) > max:
            max = len(words)
    return max


def run_lstm(train, test, max_len, seed, epoch_val, b_size):
    train, valid = sk.train_test_split(train, train_size=0.8, random_state=seed)

    X_train = train['sentence'].tolist()
    Y_train = train['label']

    X_test = test['sentence'].tolist()
    Y_test = test['label']

    X_valid = valid['sentence'].tolist()
    Y_valid = valid['label']

    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'  # out of vocabulary token
    vocab_size = 2000
    tokenizer = Tokenizer(num_words=vocab_size, char_level=False, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index
    total_words = len(word_index)

    # Padding
    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = pad_sequences(train_sequences,
                                 maxlen=max_len,
                                 padding=padding_type,
                                 truncating=trunc_type)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_padded = pad_sequences(test_sequences,
                                maxlen=max_len,
                                padding=padding_type,
                                truncating=trunc_type)
    valid_sequences = tokenizer.texts_to_sequences(X_valid)
    valid_padded = pad_sequences(valid_sequences,
                                 maxlen=max_len,
                                 padding=padding_type,
                                 truncating=trunc_type)
    print('Shape of train tensor: ', train_padded.shape)
    print('Shape of test tensor: ', test_padded.shape)
    print('Shape of valid tensor: ', valid_padded.shape)

    # Define parameter
    embedding_dim = 16
    batch_size = b_size
    epochs = epoch_val

    # Define Dense Model Architecture
    model = Sequential()
    model.add(Embedding(vocab_size,
                        embedding_dim,
                        input_length=max_len,
                        mask_zero=True))
    model.add(Bidirectional(LSTM(4, return_sequences=False))) #Bi-LSTM
    model.add(Dense(5, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(train_padded, Y_train, validation_data=(valid_padded, Y_valid), epochs=epochs, shuffle=True,
                        verbose=1, batch_size=batch_size)
    res = model.predict(test_padded)
    res = res.argmax(axis=-1)
    print(res)
    cp = skm.classification_report(Y_test.tolist(), res, output_dict=True)

    val_acc = history.history['val_accuracy'][-1]
    test_acc = cp['weighted avg']['f1-score']

    return val_acc, test_acc


# Hyperparameters
epochs = [10, 20, 30]
batch_sizes = [4, 8, 16, 32]

res_df = {"Dataset": [],
          "Seed": [],
          "Epoch": [],
          "Batch-Size": [],
          "Val-Acc": [],
          "Test-Acc": []}

# Run LSTM for each file and store results for hyperparameter combinations
train_dir = sorted(os.listdir("../training_data/test-and-training/training_data/"))
test_dir = sorted(os.listdir("../training_data/test-and-training/test_data/"))
remove_digits = str.maketrans('', '', digits)

for f in range(len(train_dir)):
    print("Experiment Number: ", f)
    name = train_dir[f].replace(".xlsx", "").replace("-train", "")
    seed = int(re.findall("\d+", name)[0])
    base_name = name.translate(remove_digits)[:-1]
    print(name), print(seed), print(base_name)

    train = pd.read_excel("../training_data/test-and-training/training_data/" + train_dir[f], index_col=False)
    test = pd.read_excel("../training_data/test-and-training/test_data/" + test_dir[f], index_col=False)
    max_len = get_max_length(train)

    for e in epochs:
        for b in batch_sizes:
            val_acc, test_acc = run_lstm(train=train, test=test, max_len=max_len,
                                         seed=seed, epoch_val=e, b_size=b)
            print(val_acc),print(test_acc)
            res_df['Dataset'].append(base_name)
            res_df['Seed'].append(seed)
            res_df['Epoch'].append(e)
            res_df['Batch-Size'].append(b)
            res_df['Val-Acc'].append(val_acc)
            res_df['Test-Acc'].append(test_acc)


print(res_df)
t = pd.DataFrame(res_df)
t.to_excel("../grid_search_results/bilstm_results/results_full.xlsx", index=False)
