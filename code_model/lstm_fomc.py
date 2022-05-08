import keras.initializers.initializers_v2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional

import tensorflow as tf
import keras

import os
import random

# -----------------------------------------------------------

df = pd.read_excel('/Users/suvanpaturi/Documents/Meeting-Minutes-Datasets/manual.xlsx')
df = df[['sentence', 'label']]
pd.set_option('display.max_rows', 500)

#Ensure seed is all the same
seed_value = 1
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def plot_history(history):

    """
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    """

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
        sentence = row['sentence']
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        words = word_tokenize(sentence.translate(str.maketrans('', '', string.punctuation)))
        if len(words) > max:
            max = len(words)
    return max


def define_vocab(df):
    text = ""
    for index, row in df.iterrows():  # format sentence for tokenization
        text += row['sentence']
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = list(set(word_tokenize(text)))
    print(tokens)
    vocab = dict((token, i + 1) for i, token in enumerate(tokens))
    return vocab


def word_vector(df, vocab_dict, max_length):
    word_vector = []
    for index, row in df.iterrows():  # format sentence for tokenization
        w_vector = []
        sentence = row['sentence']
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
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


# Data Extraction & Split
df['sentence'] = df['sentence'].str.lower()
print(df)

train, test = train_test_split(df, test_size = 0.2, random_state = 1)  # 80% Training, 20% Test split with seed of 1
train, valid = train_test_split(train, test_size = 0.2, random_state = 1)
print(train)
print(test)
data_vocab = define_vocab(df)  # build vocab from whole dataset
data_max = get_max_length(df)  # get max number of words for sentence

# input or x for training set
X_train_vector = word_vector(df=train, vocab_dict=data_vocab, max_length=data_max)
# input or x for test set
X_test_vector = word_vector(df=test, vocab_dict=data_vocab, max_length=data_max)
X_valid = word_vector(df=valid, vocab_dict=data_vocab, max_length=data_max)

Y_train_vector = [[label] for label in train['label']]  # output or y for training set
Y_test_vector = [[label] for label in test['label']]  # output or y for test set
Y_valid = [[label] for label in valid['label']]

pd.DataFrame(Y_test_vector).to_csv('y_test.csv', index=False)
pd.DataFrame(Y_train_vector).to_csv('y_train.csv', index=False)
pd.DataFrame(Y_valid).to_csv('y_valid.csv', index=False)

pd.DataFrame(X_train_vector).to_csv('x_train.csv', index=False)
pd.DataFrame(X_test_vector).to_csv('x_test.csv', index=False)
pd.DataFrame(X_valid).to_csv('x_valid.csv', index=False)

X_train_vector = pd.read_csv('x_train.csv')
X_test_vector = pd.read_csv('x_test.csv')
X_valid = pd.read_csv('x_valid.csv')

Y_train_vector = pd.read_csv('y_train.csv')
Y_test_vector = pd.read_csv('y_test.csv')
Y_valid = pd.read_csv('y_valid.csv')

# Build Model
vocab_size = len(data_vocab)  # size of vocab
print(data_max)  # max length
# each sentence is 189 values long
# ouput is one label

model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1, input_length=data_max,
                    output_dim=3,
                    embeddings_initializer = keras.initializers.initializers_v2.RandomUniform(seed=seed_value)
                    , mask_zero=True))
model.add(Dropout(rate=0.4, seed=seed_value))
model.add((LSTM(units=4, kernel_initializer = keras.initializers.initializers_v2.GlorotUniform(seed=seed_value))))
model.add(Dropout(rate=0.5, seed=seed_value))
model.add(Dense(units=100, kernel_initializer = keras.initializers.initializers_v2.RandomNormal(0,1, seed=seed_value)
                , activation='relu'))
model.add(Dropout(rate=0.6, seed=seed_value))
model.add(Dense(units=4, kernel_initializer = keras.initializers.initializers_v2.RandomNormal(0,1, seed=seed_value)
                , activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
f1_scores = {}

lstm = model.fit(X_train_vector, Y_train_vector, validation_data=[X_valid, Y_valid], epochs=25, shuffle=False)
Y_probas = model.predict(X_test_vector)
print(Y_probas)
Y_pred = Y_probas.argmax(axis=-1)
print(Y_pred)
matrix = classification_report(Y_test_vector, Y_pred, digits=6)
print(matrix)
print(plot_history(lstm))
