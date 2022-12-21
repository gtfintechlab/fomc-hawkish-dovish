import re

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

# -----------------------------------------------------------

output_dir = "../lstm_results/"
pd.set_option('display.max_rows', 500)


def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


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


def define_vocab(df):  # IGNORE for now
    text = ""
    for index, row in df.iterrows():  # format sentence for tokenization
        text += row['sentence'].lower()
    text = text.replace(",", "").replace(".", " ").replace("—", " ").replace("â€", "").replace("  ", " ") \
        .replace(";", "").replace("\n", " ")  # remove new line and separate sentences
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = list(set(word_tokenize(text)))

    vocab = dict((token, i + 1) for i, token in enumerate(set(tokens)))
    return vocab


def word_vector(df, vocab_dict, max_length):  # IGNORE for now
    word_vector = []
    for index, row in df.iterrows():  # format sentence for tokenization
        w_vector = []
        sentence = row['sentence'].replace(",", "").replace(".", " ").replace("—", " ").replace("â€", "") \
            .replace(";", "").replace("\n", " ").translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(sentence)
        for word in words:
            w_vector.append(vocab_dict.get(word, 0))
        if len(w_vector) < max_length:
            while len(w_vector) != max_length:
                w_vector.append(0)
        elif len(w_vector) >= max_length:
            w_vector = w_vector[0:max_length]
        word_vector.append(w_vector)
    return word_vector


def run_lstm(train, test, vocab_list, seed, file_path):
    train, valid = sk.train_test_split(train, train_size=0.8, random_state=seed)

    X_train = train['sentence'].tolist()
    Y_train = train['label']

    X_test = test['sentence'].tolist()
    Y_test = test['label']

    X_valid = valid['sentence'].tolist()
    Y_valid = valid['label']

    max_len = vocab_list[1]  # build vocab from whole dataset
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
    drop_value = 0.4
    # Define Dense Model Architecture
    model = Sequential()
    model.add(Embedding(vocab_size,
                        embedding_dim,
                        input_length=max_len,
                        mask_zero=True))
    model.add(LSTM(4))  # Add Bidirectional to convert to bilstm
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_value))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']
                  )
    model.summary()
    model.fit(train_padded, Y_train, validation_data=(valid_padded, Y_valid), epochs=10, shuffle=True, verbose=1)
    res = model.predict(test_padded)
    res = res.argmax(axis=-1)
    print(res)

    temp = pd.DataFrame({'sentence': X_test, 'label': Y_test.tolist(), "pred-label": res})
    temp.to_excel(file_path, index=False)
    cp = skm.classification_report(Y_test.tolist(), res, output_dict=True)
    return cp['weighted avg']['f1-score']


# Run LSTM for each file
vocab_list = {}
df_dir = os.listdir("../labeled_data")
print(df_dir)
for f_name in df_dir:
    if not f_name.startswith("."):
        file_name = f_name.replace(".xlsx", "")
        df = pd.read_excel("../labeled_data/" + f_name)
        vocab_list[file_name] = (define_vocab(df), get_max_length(df))
train_dir = sorted(os.listdir("../training_data/"))
test_dir = sorted(os.listdir("../test_data/"))

remove_digits = str.maketrans('', '', digits)
score_dict = {}

for f in range(len(train_dir)):
    name = train_dir[f].replace(".xlsx", "").replace("-train", "")
    seed = int(re.findall("\d+", name)[0])
    base_name = name.translate(remove_digits)[:-1]
    print(name), print(seed)

    train = pd.read_excel("../training_data/" + train_dir[f], index_col=False)
    test = pd.read_excel("../test_data/" + test_dir[f], index_col=False)

    val = run_lstm(train=train, test=test, vocab_list=vocab_list[base_name], seed=seed,
                   file_path=output_dir + base_name + '-results-' + str(seed) + ".xlsx")

    if base_name in score_dict:
        score_dict[base_name] += val
    else:
        score_dict[base_name] = val

for s in score_dict:
    score_dict[s] /= 3
print(score_dict)
