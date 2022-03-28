import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import string

import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import keras
from keras import Model, metrics
import keras.engine as KE
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

# -----------------------------------------------------------
df = pd.read_excel('../training_data/manual_v2.xlsx')
df = df[['sentence', 'label']]
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
        sentence = row['sentence'].replace(",", " ").replace(".", " ") \
            .replace(";", " ").replace("\n", "").translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(sentence)
        if len(words) > max:
            max = len(words)
    return max


def define_vocab(df):
    text = ""
    for index, row in df.iterrows():  # format sentence for tokenization
        text += row['sentence']
    text = text.replace(",", "").replace(".", " ") \
        .replace(";", "").replace("\n", "")  # remove new line and separate sentences
    text = text.translate(str.maketrans('', '', string.punctuation))
    # print(text)
    tokens = list(set(word_tokenize(text)))
    # print(tokens)

    vocab = dict((token, i + 1) for i, token in enumerate(tokens))
    # print(vocab)
    return vocab


def word_vector(df, vocab_dict, max_length):
    word_vector = []
    for index, row in df.iterrows():  # format sentence for tokenization
        w_vector = []
        sentence = row['sentence'].replace(",", " ").replace(".", " ") \
            .replace(";", " ").replace("\n", " ").translate(str.maketrans('', '', string.punctuation))
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
train, test = train_test_split(df, test_size=0.2, random_state=1)  # 80% Training, 20% Test split with seed of 1
train, valid = train_test_split(train, test_size=0.2, random_state=1)
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

pd.DataFrame(Y_test_vector).to_csv('../model_data/lstm_training_files/y_test.csv', index=False)
pd.DataFrame(Y_train_vector).to_csv('../model_data/lstm_training_files/y_train.csv', index=False)
pd.DataFrame(Y_valid).to_csv('../model_data/lstm_training_files/y_valid.csv', index=False)

pd.DataFrame(X_train_vector).to_csv('../model_data/lstm_training_files/x_train.csv', index=False)
pd.DataFrame(X_test_vector).to_csv('../model_data/lstm_training_files/x_test.csv', index=False)
pd.DataFrame(X_valid).to_csv('../model_data/lstm_training_files/x_valid.csv', index=False)

X_train_vector = pd.read_csv('../model_data/lstm_training_files/x_train.csv')
X_test_vector = pd.read_csv('../model_data/lstm_training_files/x_test.csv')
X_valid = pd.read_csv('../model_data/lstm_training_files/x_valid.csv')

Y_train_vector = pd.read_csv('../model_data/lstm_training_files/y_train.csv')
Y_test_vector = pd.read_csv('../model_data/lstm_training_files/y_test.csv')
Y_valid = pd.read_csv('../model_data/lstm_training_files/y_valid.csv')

## one-hot encoding at output layer
num_classes = 3
Y_train_vector = to_categorical(Y_train_vector, num_classes)
Y_test_vector = to_categorical(Y_test_vector, num_classes)
Y_valid = to_categorical(Y_valid, num_classes)


# Build Model
vocab_size = len(data_vocab)  # size of vocab
# each sentence is 189 values long
# ouput is one label

model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1, input_length=data_max, output_dim=3, mask_zero=True))
model.add(Dropout(rate=0.2))
model.add(LSTM(units=4))
model.add(Dropout(rate=0.2))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lstm = model.fit(X_train_vector, Y_train_vector, validation_data = [X_valid, Y_valid], epochs=20, verbose=1)
Y_probas = model.predict(X_test_vector)

matrix = confusion_matrix(Y_test_vector.argmax(axis=1), Y_probas.argmax(axis=1))
print(matrix)

target_names = ["Dovish", "Hawkish", "Neutral"]
print(classification_report(Y_test_vector.argmax(axis=1), Y_probas.argmax(axis=1), target_names=target_names, digits=4))

print(plot_history(lstm))
