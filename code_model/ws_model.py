# !pip install snorkel

# Libraries
import nltk
nltk.download('punkt')
import pandas as pd
import numpy as np
import logging
import spacy
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
nlp = spacy.load('en_core_web_sm')
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

## Using our annotation rules
A1_o = ["trade deficit", "economic growth", "demand rate", "inflation", "economy", "energy cost", "price", "treasury securities", "funds rate", "fund rate", "money supply","fed", "bond"]
A2_o = ["decelerate", "depreciate", "low", "decrease", "fall", "decline", "subpar", "buy"]

B1_o = ["dollar", "unemployment", "loan"]
B2_o = ["increase", "high", "rise", "accelerate", "ease", "expand", "upward", "improve", "rapid", "appreciate", "sell", "widen"]

N = ["tightened on balance", "stability", "sustainable", " wait for more evidence", "sustained", "stabilization", "productivity improvement",
           "mixed", "flattening", "moderate", "reaffirmed", "eased off", "remain subdued", "transitory", "offset"]

## Yuriy Paper
A1 = ["inflation expectation", "interest rate", "bank rate", "fund rate", "price", "economic activity", "inflation", "employment"]
A2 = ["anchor", "cut", "subdue", "decline", "decrease", "reduce", "low", "drop", "fall", "fell", "decelarate", "slow", "pause", "pausing",
      "stable", "non-accelerating", "downward", "tighten"]

B1 = ["unemployment", "growth", "exchange rate", "productivity", "deficit", "demand", "job market", "monetary policy"]
B2 = ["ease", "easing", "rise", "rising", "increase", "expand", "improve", "strong", "upward", "raise", "high", "rapid"]

C = ["weren't", "were not", "wasn't", "was not", 'did not', "didn't", "do not", "don't", 'will not', "won't"]

## Modular functions
def func_B1_o(x):
    s = x.lower()
    for word in B1_o:
        if word in s:
            return 1
    return 0

def func_B2_o(x):
  words = nlp(x.lower())
  arr_lemma = [(token,token.lemma_) for token in words]
  for word in arr_lemma:
      if word[1] in B2_o:
          return 1
  return 0

def func_A1_o(x):
    s = x.lower()
    for word in A1_o:
        if word in s:
            return 1
    return 0

def func_A2_o(x):
    words = nlp(x.lower())
    arr_lemma = [(token,token.lemma_) for token in words]
    for word in arr_lemma:
        if word[1] in A2_o:
            return 1
    return 0
    

def func_A1(x):
    s = x.lower()
    for word in A1:
        if word in s:
            return 1
    return 0

def func_A2(x):
    s = x.lower()
    for word in A2:
        if word in s:
            return 1
    return 0

def func_B1(x):
    s = x.lower()
    for word in B1:
        if word in s:
            return 1
    return 0

def func_B2(x):
    s = x.lower()
    for word in B2:
        if word in s:
            return 1
    return 0

def func_negation(x):
  s = x.lower()
  for word in C:
    if word in s:
      return True
  return False

## Labelling functions
@labeling_function()
def func_neutral(x):
  s = x.sentence.lower()
  for word in N:
    if word in s:
      return 2
  return -1

@labeling_function()
def func_a2_b1(x):
  s = x.sentence
  b1 = func_B1(s)
  a2 = func_A2(s)
  if b1==1 and a2==1:
    neg = func_negation(s)
    if neg:
      return 0
    return 1
  return -1

@labeling_function()
def func_a1_b2(x):
  s = x.sentence
  a1 = func_A1(s)
  b2 = func_B2(s)
  neg = func_negation(s)
  if a1==1 and b2==1:
    if neg:
      return 0
    return 1
  return -1


@labeling_function()
def func_a1_a2(x):
  s = x.sentence
  a2 = func_A2(s)
  a1 = func_A1(s)
  neg = func_negation(s)
  if a2==1 and a1==1:
    if neg:
      return 1
    return 0
  return -1

@labeling_function()
def func_b1_b2(x):
  s = x.sentence
  b1 = func_B1(s)
  b2 = func_B2(s)
  neg = func_negation(s)
  if b1==1 and b2==1:
    if neg:
      return 1
    return 0
  return -1

@labeling_function()
def func_a2_b1_o(x):
  s = x.sentence
  b1 = func_B1_o(s)
  a2 = func_A2_o(s)
  if b1==1 and a2==1:
    neg = func_negation(s)
    if neg:
      return 0
    return 1
  return -1

@labeling_function()
def func_a1_b2_o(x):
  s = x.sentence
  a1 = func_A1_o(s)
  b2 = func_B2_o(s)
  neg = func_negation(s)
  if a1==1 and b2==1:
    if neg:
      return 0
    return 1
  return -1


@labeling_function()
def func_a1_a2_o(x):
  s = x.sentence
  a2 = func_A2_o(s)
  a1 = func_A1_o(s)
  neg = func_negation(s)
  if a2==1 and a1==1:
    if neg:
      return 1
    return 0
  return -1

@labeling_function()
def func_b1_b2_o(x):
  s = x.sentence
  b1 = func_B1_o(s)
  b2 = func_B2_o(s)
  neg = func_negation(s)
  if b1==1 and b2==1:
    if neg:
      return 1
    return 0
  return -1

## Driver function for labelling data
def label_Data():
    df_train = pd.read_excel("manual_combined.xlsx")
    # # WS1
    # lfs = [func_a1_b2, func_b1_b2, func_a1_a2, func_a2_b1]

    # # WS2
    # lfs = [func_a1_b2_o, func_b1_b2_o, func_a1_a2_o, func_a2_b1_o, func_neutral]
    
    # WS3
    lfs = [func_a1_b2,func_a1_b2_o, func_b1_b2, func_b1_b2_o,  func_a1_a2, func_a1_a2_o, func_a2_b1, func_a2_b1_o, func_neutral]
    applier = PandasLFApplier(lfs=lfs)

    # Pass data for getting the labels
    L_train = applier.apply(df=df_train)
    status = []
    print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())
    
    for i in range(len(L_train)):
      l = list(L_train[i])
      if (l.count(2)!=0 and l.count(-1)==8) or l.count(-1)==9 or (l.count(0)==l.count(1) and l.count(0)>0):
        status.append(2)
      elif l.count(0)>l.count(1):
        status.append(0)
      elif l.count(1)>l.count(0):
        status.append(1)
      else:
        status.append(2)        
    df_train['model_label'] = status
    df_train.to_excel("labelled_data.xlsx")
    return L_train, df_train['label'].tolist(), status


L_train, manual_labels, status = label_Data()
print(classification_report(manual_labels, status))
print(accuracy_score(manual_labels, status))