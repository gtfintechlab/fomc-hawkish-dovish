import numpy as np
import time
import multiprocessing
import itertools
import csv
import os
import os.path
import spacy
from collections import Counter
from datetime import datetime
import pandas as pd
import operator
import getpass
from nltk.corpus import sentiwordnet as swn  
from pathlib import Path
import re 
#import senticnet5
import senti_bignomics


def safe_string_cast_to_numerictype(val, to_type, default = None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


def FeelIt(tlemma, tpos=None, tokentlemma=None, PrintScr=False, UseSenticNet=False):
    tosearcsenticnet = tlemma.lower().replace(" ", "_")
    sentibignomicsitem = senti_bignomics.senti_bignomics.get(tosearcsenticnet)
    if sentibignomicsitem and sentibignomicsitem[0]:
        valstr = sentibignomicsitem[0]
        senti_bignomics_sentiment = safe_string_cast_to_numerictype(valstr, float, 0)
        computed_sentiment = senti_bignomics_sentiment
        return computed_sentiment
    if tpos == "NOUN":
        posval = "n"
    elif tpos == "VERB":
        posval = "v"
    elif tpos == "ADJ":
        posval = "a"
    elif tpos == "ADV":
        posval = "r"
    else:
        posval = "n"
    avgsc = 0
    try:
        llsenses_pos = list(swn.senti_synsets(tlemma.lower(), posval))
    except:
        llsenses_pos = []
    if llsenses_pos and len(llsenses_pos) > 0:
        for thistimescore in llsenses_pos:
            avgsc = thistimescore.pos_score() - thistimescore.neg_score()
            if avgsc != 0:
                break
    if avgsc == 0 and posval == "a":
        posval = "s"
        try:
            llsenses_pos = list(swn.senti_synsets(tlemma.lower(), posval))
        except:
            llsenses_pos = []
        if llsenses_pos and len(llsenses_pos) > 0:
            for thistimescore in llsenses_pos:
                avgsc = thistimescore.pos_score() - thistimescore.neg_score()
                if avgsc != 0:
                    break
    sentic_sentiment = 0
    if UseSenticNet == True:
        tosearcsenticnet = tlemma.lower().replace(" ", "_")
        senticitem = senticnet5.senticnet.get(tosearcsenticnet)  
        if senticitem and senticitem[7]:
            valstr = senticitem[7]
            sentic_sentiment = safe_string_cast_to_numerictype(valstr, float, 0)
    if sentic_sentiment != 0:
        computed_sentiment = (sentic_sentiment + avgsc) / 2
    else:
        computed_sentiment = avgsc
    return computed_sentiment


def FeelIt_OverallSentiment(toi, PrintScr=False, UseSenticNet=False):
    sentim_all = 0.0
    countsss = 0
    for xin in toi.sent:
        if (xin.pos_ == "ADJ") | (xin.pos_ == "ADV") | (xin.pos_ == "NOUN") | (xin.pos_ == "PROPN") | (
                xin.pos_ == "VERB"):
            sentim_app = FeelIt(xin.lemma_.lower(), xin.pos_, xin)
            if sentim_app != 0.0:
                countsss = countsss + 1
            sentim_all = sentim_all + sentim_app
    if countsss > 0:
        sentim_all = sentim_all / countsss
    return sentim_all


def PREP_token_IE_parsing(xin,singleNOUNs, singleCompoundedHITs, singleCompoundedHITs_toEXCLUDE,LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP, COMPUTE_OVERALL_SENTIMENT_SCORE, minw, maxw, FoundVerb, t, nountoskip=None, previousprep=None):
    listOfPreps = []
    listOfPreps_sentim = []
    lxin_n = [x for x in xin.lefts]
    rxin_n = [x for x in xin.rights]
    if lxin_n:
        for xinxin in lxin_n:
            if xinxin.dep_ == "pobj" and xinxin.pos_ == "NOUN" and IsInterestingToken(
                    xinxin) and xinxin.lemma_.lower() != t.lemma_.lower():
                if (nountoskip):
                    if xinxin.lemma_.lower() == nountoskip.lemma_.lower():
                        continue
                minw = min(minw, xinxin.i)
                maxw = max(maxw, xinxin.i)
                other_list_NOUN, sentilist, minw, maxw = NOUN_token_IE_parsing(xinxin,singleNOUNs,singleCompoundedHITs, singleCompoundedHITs_toEXCLUDE,LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                               VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE, minw=minw,maxw=maxw,
                                                                               verbtoskip=FoundVerb, nountoskip=t)
                sentim_noun = FeelIt(xinxin.lemma_.lower(), xinxin.pos_, xinxin)
                if other_list_NOUN and len(other_list_NOUN) > 0:
                    listNoun_app = []
                    FoundNounInlist = "___" + xinxin.lemma_.lower() + " ["+xinxin.pos_+", "+xinxin.tag_+" ("+str(sentim_noun)+ ")]"
                    for modin in other_list_NOUN:
                        FoundNounInlist = FoundNounInlist + "___" + modin
                    listNoun_app.append(FoundNounInlist)
                    other_list_NOUN = listNoun_app
                    
                    listOfPreps.extend(other_list_NOUN)
                else:
                    FoundNounInlist = "___" + xinxin.lemma_.lower() + " ["+xinxin.pos_+", "+xinxin.tag_+" ("+str(sentim_noun)+ ")]" + "___"
                    listOfPreps.append(FoundNounInlist)
                if sentilist and len(sentilist) > 0:
                    for sentin in sentilist:
                        if sentin != 0:
                            if sentim_noun == 0:
                                sentim_noun = sentin
                            else:
                                if (sentim_noun > 0 and sentin < 0) or (
                                        sentim_noun < 0 and sentin > 0):  
                                    sentim_noun = np.sign(sentin) * np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin)) 
                                else:  
                                    sentim_noun = np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  
                listOfPreps_sentim.append(sentim_noun)
            elif xinxin.dep_ == "pcomp" and xinxin.pos_ == "VERB" and xinxin.lemma_.lower() != t.lemma_.lower():  
                minw = min(minw, xinxin.i)
                maxw = max(maxw, xinxin.i)
                iterated_list_VERB, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(xinxin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                              LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                              t, minw, maxw,nountoskip=nountoskip,previousverb=FoundVerb)
                if iterated_list_VERB and len(iterated_list_VERB) > 0:
                    listOfPreps.extend(iterated_list_VERB)
                    if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                        listOfPreps_sentim.extend(list_verbs_sentim_app)
            elif xinxin.dep_ == "prep" and xinxin.pos_ == "ADP":
                if (previousprep):
                    if xinxin.lemma_.lower() == previousprep.lemma_.lower():
                        continue
                minw = min(minw, xinxin.i)
                maxw = max(maxw, xinxin.i)
                iterated_list_prep, iterated_list_prep_sentim, minw, maxw = PREP_token_IE_parsing(xinxin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                                  minw=minw,maxw=maxw,FoundVerb=FoundVerb,t=t,nountoskip=nountoskip,
                                                                                                  previousprep=xin)
                if iterated_list_prep and len(iterated_list_prep) > 0:
                    listOfPreps.extend(iterated_list_prep)
                    if iterated_list_prep_sentim and len(iterated_list_prep_sentim) > 0:
                        listOfPreps_sentim.extend(iterated_list_prep_sentim)
    if rxin_n:
        for xinxin in rxin_n:
            if xinxin.dep_ == "pobj" and xinxin.pos_ == "NOUN" and IsInterestingToken(
                    xinxin) and xinxin.lemma_.lower() != t.lemma_.lower():
                if (nountoskip):
                    if xinxin.lemma_.lower() == nountoskip.lemma_.lower():
                        continue
                minw = min(minw, xinxin.i)
                maxw = max(maxw, xinxin.i)
                other_list_NOUN, sentilist, minw, maxw = NOUN_token_IE_parsing(xinxin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                               LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                               minw=minw, maxw=maxw,verbtoskip=FoundVerb, nountoskip=t)
                sentim_noun = FeelIt(xinxin.lemma_.lower(), xinxin.pos_, xinxin)
                if other_list_NOUN and len(other_list_NOUN) > 0:
                    listNoun_app = []
                    FoundNounInlist = "___" + xinxin.lemma_.lower() + " ["+xinxin.pos_+", "+xinxin.tag_+" ("+str(sentim_noun)+ ")]"
                    for modin in other_list_NOUN:
                        FoundNounInlist = FoundNounInlist + "___" + modin
                    listNoun_app.append(FoundNounInlist)
                    other_list_NOUN = listNoun_app
                    #
                    listOfPreps.extend(other_list_NOUN)
                else:
                    FoundNounInlist = "___" + xinxin.lemma_.lower() + " ["+xinxin.pos_+", "+xinxin.tag_+" ("+str(sentim_noun)+ ")]" + "___"
                    listOfPreps.append(FoundNounInlist)
                if sentilist and len(sentilist) > 0:
                    for sentin in sentilist:
                        if sentin != 0:
                            if sentim_noun == 0:
                                sentim_noun = sentin
                            else:
                                if (sentim_noun > 0 and sentin < 0) or (
                                        sentim_noun < 0 and sentin > 0):  
                                    sentim_noun = np.sign(sentin) * np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  
                                else:  
                                    sentim_noun = np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin)) 
                listOfPreps_sentim.append(sentim_noun)
            elif xinxin.dep_ == "pcomp" and xinxin.pos_ == "VERB" and xinxin.lemma_.lower() != t.lemma_.lower():
                minw = min(minw, xinxin.i)
                maxw = max(maxw, xinxin.i)
                iterated_list_VERB, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(xinxin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                              LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                              t, minw, maxw,nountoskip=nountoskip,previousverb=FoundVerb)
                if iterated_list_VERB and len(iterated_list_VERB) > 0:
                    listOfPreps.extend(iterated_list_VERB)
                    if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                        listOfPreps_sentim.extend(list_verbs_sentim_app)
            elif xinxin.dep_ == "prep" and xinxin.pos_ == "ADP":
                if (previousprep):
                    if xinxin.lemma_.lower() == previousprep.lemma_.lower():
                        continue
                minw = min(minw, xinxin.i)
                maxw = max(maxw, xinxin.i)
                iterated_list_prep, iterated_list_prep_sentim, minw, maxw = PREP_token_IE_parsing(xinxin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                                  minw=minw,maxw=maxw,FoundVerb=FoundVerb,t=t,nountoskip=nountoskip,
                                                                                                  previousprep=xin)
                if iterated_list_prep and len(iterated_list_prep) > 0:
                    listOfPreps.extend(iterated_list_prep)
                    if iterated_list_prep_sentim and len(iterated_list_prep_sentim) > 0:
                        listOfPreps_sentim.extend(iterated_list_prep_sentim)
    return listOfPreps, listOfPreps_sentim, minw, maxw


def VERB_token_IE_parsing(FoundVerb, singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE, t, minw, maxw, nountoskip=None,previousverb=None):
    listVerbs = []
    listVerbs_sentim = []
    CompoundsOfSingleHit = findCompoundedHITsOfTerm(singleCompoundedHITs, FoundVerb)
    FoundNeg = None
    FoundVerbAdverb = ""
    FoundVerbAdverb_sentim = 0
    listFoundModofVB = []
    listFoundModofVB_sentim = []
    l_n = [x for x in FoundVerb.lefts]
    if l_n:
        for xin in l_n:
            lxin_n = [x for x in xin.lefts]
            rxin_n = [x for x in xin.rights]
            if xin.dep_ == "neg":
                FoundNeg = "__not"
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
            elif xin.dep_ == "advmod" and (xin.pos_ == "ADV" and (xin.tag_ == "RBS" or xin.tag_ == "RBR")):
                if (xin.lemma_.lower() in CompoundsOfSingleHit):
                    continue
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                sentim_app = FeelIt(xin.lemma_.lower(), xin.pos_, xin)
                FoundVerbAdverb = FoundVerbAdverb + "__" + xin.lemma_.lower() + " [" + xin.pos_ + ", " + xin.tag_ + " (" + str(sentim_app) + ")]"
                if FoundVerbAdverb_sentim == 0:
                    FoundVerbAdverb_sentim = sentim_app
                else:
                    if (FoundVerbAdverb_sentim > 0 and sentim_app < 0) or (FoundVerbAdverb_sentim < 0 and sentim_app > 0):
                        FoundVerbAdverb_sentim = FoundVerbAdverb_sentim + sentim_app
                    else:  
                        FoundVerbAdverb_sentim = np.sign(FoundVerbAdverb_sentim) * (
                                abs(FoundVerbAdverb_sentim) + (1 - abs(FoundVerbAdverb_sentim)) * abs(sentim_app))
            elif (xin.dep_ == "acomp" or xin.dep_ == "oprd") and (xin.pos_ == "ADJ" and (
                    xin.tag_ == "JJR" or xin.tag_ == "JJS" or xin.tag_ == "JJ")) and xin.lemma_.lower() != t.lemma_.lower(): 
                if xin.lemma_.lower() in CompoundsOfSingleHit:
                    continue
                foundadv = ""
                foundadv_sentim = 0
                if lxin_n:
                    for xinxin in lxin_n:
                        if ((xinxin.dep_ == "advmod" and (xinxin.pos_ == "ADV" and (
                                xinxin.tag_ == "RBS" or xinxin.tag_ == "RBR"))) or (
                                    xinxin.dep_ == "conj" and (xinxin.pos_ == "ADJ" and (
                                    xinxin.tag_ == "JJR" or xinxin.tag_ == "JJS" or xinxin.tag_ == "JJ")))) and xinxin.lemma_.lower() != t.lemma_.lower():  
                            minw = min(minw, xinxin.i)
                            maxw = max(maxw, xinxin.i)
                            sentim_app = FeelIt(xinxin.lemma_.lower(), xinxin.pos_, xinxin)
                            foundadv = foundadv + "__" + xinxin.lemma_.lower() + " [" + xinxin.pos_ + ", " + xinxin.tag_ + " (" + str(sentim_app) + ")]"
                            if foundadv_sentim == 0:
                                foundadv_sentim = sentim_app
                            else:
                                if (foundadv_sentim > 0 and sentim_app < 0) or (foundadv_sentim < 0 and sentim_app > 0):
                                    foundadv_sentim = foundadv_sentim + sentim_app
                                else:  
                                    foundadv_sentim = np.sign(foundadv_sentim) * (
                                            abs(foundadv_sentim) + (1 - abs(foundadv_sentim)) * abs(sentim_app))
                if rxin_n:
                    for xinxin in rxin_n:
                        if ((xinxin.dep_ == "advmod" and (xinxin.pos_ == "ADV" and (
                                xinxin.tag_ == "RBS" or xinxin.tag_ == "RBR"))) or (
                                    xinxin.dep_ == "conj" and (xinxin.pos_ == "ADJ" and (
                                    xinxin.tag_ == "JJR" or xinxin.tag_ == "JJS" or xinxin.tag_ == "JJ")))) and xinxin.lemma_.lower() != t.lemma_.lower():  
                            minw = min(minw, xinxin.i)
                            maxw = max(maxw, xinxin.i)
                            sentim_app = FeelIt(xinxin.lemma_.lower(), xinxin.pos_, xinxin)
                            foundadv = foundadv + "__" + xinxin.lemma_.lower() + " [" + xinxin.pos_ + ", " + xinxin.tag_ + " (" + str(sentim_app) + ")]"
                            if foundadv_sentim == 0:
                                foundadv_sentim = sentim_app
                            else:
                                if (foundadv_sentim > 0 and sentim_app < 0) or (
                                        foundadv_sentim < 0 and sentim_app > 0):  
                                    foundadv_sentim = foundadv_sentim + sentim_app
                                else:  
                                    foundadv_sentim = np.sign(foundadv_sentim) * (
                                            abs(foundadv_sentim) + (1 - abs(foundadv_sentim)) * abs(sentim_app))
                sentim_compound = FeelIt(xin.lemma_.lower(), xin.pos_, xin)
                listFoundModofVB.append((xin.lemma_.lower() + " [" + xin.pos_ + ", " + xin.tag_ + " (" + str(sentim_compound) + ")]" + foundadv))
                if sentim_compound == 0:
                    sentim_compound = foundadv_sentim
                else:
                    if (foundadv_sentim > 0 and sentim_app < 0) or (
                            foundadv_sentim < 0 and sentim_app > 0):  
                        sentim_compound = np.sign(foundadv_sentim) * np.sign(sentim_compound) * (
                                abs(sentim_compound) + (1 - abs(sentim_compound)) * abs(
                            foundadv_sentim))  
                    else:  
                        sentim_compound = np.sign(sentim_compound) * (
                                abs(sentim_compound) + (1 - abs(sentim_compound)) * abs(foundadv_sentim))  
                listFoundModofVB_sentim.append(sentim_compound)
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
            elif (xin.dep_ == "dobj" or xin.dep_ == "attr") and xin.pos_ == "NOUN" and IsInterestingToken(
                    xin) and xin.lemma_.lower() != t.lemma_.lower():
                if (nountoskip):
                    if xin.lemma_.lower() == nountoskip.lemma_.lower():
                        continue
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                iterated_list_NOUN, sentilist, minw, maxw = NOUN_token_IE_parsing(xin, singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                  VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE, minw=minw, maxw=maxw,
                                                                                  verbtoskip=FoundVerb, nountoskip=t)
                sentim_noun = FeelIt(xin.lemma_.lower(), xin.pos_, xin)
                if iterated_list_NOUN and len(iterated_list_NOUN) > 0:
                    #
                    listNoun_app = []
                    for modin in iterated_list_NOUN:
                        FoundNounInlist = "___" + xin.lemma_.lower() + " [" + xin.pos_ + ", " + xin.tag_ + " (" + str(
                            sentim_noun) + ")]" + "___" + modin
                        listNoun_app.append(FoundNounInlist)
                    iterated_list_NOUN = listNoun_app
                    #
                    listFoundModofVB.extend(iterated_list_NOUN)
                else:
                    FoundNounInlist = "___" + xin.lemma_.lower() + " [" + xin.pos_ + ", " + xin.tag_ + " (" + str(
                        sentim_noun) + ")]" + "___"
                    listFoundModofVB.append(FoundNounInlist)
                if sentilist and len(sentilist) > 0:
                    for sentin in sentilist:
                        if sentin != 0:
                            if sentim_noun == 0:
                                sentim_noun = sentin
                            else:
                                if (sentim_noun > 0 and sentin < 0) or (
                                        sentim_noun < 0 and sentin > 0):  
                                    sentim_noun = np.sign(sentin) * np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  
                                else: 
                                    sentim_noun = np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin)) 
                listFoundModofVB_sentim.append(sentim_noun)
            elif xin.dep_ == "prep" and xin.pos_ == "ADP" and xin.lemma_.lower() != t.lemma_.lower():
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                iterated_list_prep, iterated_list_prep_sentim, minw, maxw = PREP_token_IE_parsing(xin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                                  minw=minw,maxw=maxw,FoundVerb=FoundVerb,t=t,nountoskip=nountoskip)
                if iterated_list_prep and len(iterated_list_prep) > 0:
                    listFoundModofVB.extend(iterated_list_prep)
                    if iterated_list_prep_sentim and len(iterated_list_prep_sentim) > 0:
                        listFoundModofVB_sentim.extend(iterated_list_prep_sentim)
            if FoundNeg is None:
                for xinxin in lxin_n:
                    if (xinxin.dep_ == "neg"):
                        FoundNeg = "__not"
                        minw = min(minw, xinxin.i)
                        maxw = max(maxw, xinxin.i)
                for xinxin in rxin_n:
                    if (xinxin.dep_ == "neg"):
                        FoundNeg = "__not"
                        minw = min(minw, xinxin.i)
                        maxw = max(maxw, xinxin.i)
    l_r = [x for x in FoundVerb.rights]
    if l_r:
        for xin in l_r:
            lxin_n = [x for x in xin.lefts]
            rxin_n = [x for x in xin.rights]
            if xin.dep_ == "neg":
                FoundNeg = "__not"
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
            elif xin.dep_ == "advmod" and (xin.pos_ == "ADV" and (xin.tag_ == "RBS" or xin.tag_ == "RBR")):
                if (xin.lemma_.lower() in CompoundsOfSingleHit):
                    continue
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                sentim_app = FeelIt(xin.lemma_.lower(), xin.pos_, xin)
                FoundVerbAdverb = FoundVerbAdverb + "__" + xin.lemma_.lower() + " ["+xin.pos_+", "+xin.tag_+" ("+str(sentim_app)+ ")]"
                if FoundVerbAdverb_sentim == 0:
                    FoundVerbAdverb_sentim = sentim_app
                else:
                    if (FoundVerbAdverb_sentim > 0 and sentim_app < 0) or (FoundVerbAdverb_sentim < 0 and sentim_app > 0):
                        FoundVerbAdverb_sentim = FoundVerbAdverb_sentim + sentim_app
                    else:  
                        FoundVerbAdverb_sentim = np.sign(FoundVerbAdverb_sentim) * (
                                abs(FoundVerbAdverb_sentim) + (1 - abs(FoundVerbAdverb_sentim)) * abs(sentim_app))
            elif (xin.dep_ == "acomp" or xin.dep_ == "oprd") and (xin.pos_ == "ADJ" and (
                    xin.tag_ == "JJR" or xin.tag_ == "JJS" or xin.tag_ == "JJ")) and xin.lemma_.lower() != t.lemma_.lower():  
                if  xin.lemma_.lower() in CompoundsOfSingleHit:
                    continue
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                foundadv = ""
                foundadv_sentim = 0
                if lxin_n:
                    for xinxin in lxin_n:
                        if ((xinxin.dep_ == "advmod" and (xinxin.pos_ == "ADV" and (
                                xinxin.tag_ == "RBS" or xinxin.tag_ == "RBR"))) or (
                                    xinxin.dep_ == "conj" and (xinxin.pos_ == "ADJ" and (
                                    xinxin.tag_ == "JJR" or xinxin.tag_ == "JJS" or xinxin.tag_ == "JJ")))) and xinxin.lemma_.lower() != t.lemma_.lower():  
                            minw = min(minw, xinxin.i)
                            maxw = max(maxw, xinxin.i)
                            sentim_app = FeelIt(xinxin.lemma_.lower(), xinxin.pos_, xinxin)
                            foundadv = foundadv + "__" + xinxin.lemma_.lower() + " [" + xinxin.pos_ + ", " + xinxin.tag_ + " (" + str(sentim_app) + ")]"
                            if foundadv_sentim == 0:
                                foundadv_sentim = sentim_app
                            else:
                                if (foundadv_sentim > 0 and sentim_app < 0) or (
                                        foundadv_sentim < 0 and sentim_app > 0):  
                                    foundadv_sentim = foundadv_sentim + sentim_app
                                else:  
                                    foundadv_sentim = np.sign(foundadv_sentim) * (
                                            abs(foundadv_sentim) + (1 - abs(foundadv_sentim)) * abs(sentim_app))
                if rxin_n:
                    for xinxin in rxin_n:
                        if ((xinxin.dep_ == "advmod" and (xinxin.pos_ == "ADV" and (
                                xinxin.tag_ == "RBS" or xinxin.tag_ == "RBR"))) or (
                                    xinxin.dep_ == "conj" and (xinxin.pos_ == "ADJ" and (
                                    xinxin.tag_ == "JJR" or xinxin.tag_ == "JJS" or xinxin.tag_ == "JJ")))) and xinxin.lemma_.lower() != t.lemma_.lower():  
                            minw = min(minw, xinxin.i)
                            maxw = max(maxw, xinxin.i)
                            sentim_app = FeelIt(xinxin.lemma_.lower(), xinxin.pos_, xinxin)
                            foundadv = foundadv + "__" + xinxin.lemma_.lower() + " ["+xinxin.pos_+", "+xinxin.tag_+" ("+str(sentim_app)+ ")]"
                            if foundadv_sentim == 0:
                                foundadv_sentim = sentim_app
                            else:
                                if (foundadv_sentim > 0 and sentim_app < 0) or (
                                        foundadv_sentim < 0 and sentim_app > 0):  
                                    foundadv_sentim = foundadv_sentim + sentim_app
                                else:  
                                    foundadv_sentim = np.sign(foundadv_sentim) * (
                                            abs(foundadv_sentim) + (1 - abs(foundadv_sentim)) * abs(sentim_app))
                sentim_compound = FeelIt(xin.lemma_.lower(), xin.pos_, xin)
                listFoundModofVB.append((xin.lemma_.lower() + " [" + xin.pos_ + ", " + xin.tag_ + " (" + str(sentim_compound) + ")]" + foundadv))
                if sentim_compound == 0:
                    sentim_compound = foundadv_sentim
                else:
                    if (foundadv_sentim > 0 and sentim_compound < 0) or (foundadv_sentim < 0 and sentim_compound > 0):
                        sentim_compound = np.sign(foundadv_sentim) * np.sign(sentim_compound) * (
                                abs(sentim_compound) + (1 - abs(sentim_compound)) * abs(
                            foundadv_sentim))  
                    else: 
                        sentim_compound = np.sign(sentim_compound) * (
                                abs(sentim_compound) + (1 - abs(sentim_compound)) * abs(foundadv_sentim))  
                listFoundModofVB_sentim.append(sentim_compound)
            elif (xin.dep_ == "dobj" or xin.dep_ == "attr") and xin.pos_ == "NOUN" and IsInterestingToken(
                    xin) and xin.lemma_.lower() != t.lemma_.lower():
                if nountoskip:
                    if xin.lemma_.lower() == nountoskip.lemma_.lower():
                        continue
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                iterated_list_NOUN, sentilist, minw, maxw = NOUN_token_IE_parsing(xin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                  VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE, minw=minw, maxw=maxw,
                                                                                  verbtoskip=FoundVerb, nountoskip=t)
                sentim_noun = FeelIt(xin.lemma_.lower(), xin.pos_, xin)
                if iterated_list_NOUN and len(iterated_list_NOUN) > 0:
                    #
                    listNoun_app = []
                    for modin in iterated_list_NOUN:
                        FoundNounInlist = "___" + xin.lemma_.lower() + " ["+xin.pos_+", "+xin.tag_+" ("+str(sentim_noun)+ ")]" + "___" + modin
                        listNoun_app.append(FoundNounInlist)
                    iterated_list_NOUN = listNoun_app
                    #
                    listFoundModofVB.extend(iterated_list_NOUN)
                else:
                    FoundNounInlist = "___" + xin.lemma_.lower() + " ["+xin.pos_+", "+xin.tag_+" ("+str(sentim_noun)+ ")]" + "___"
                    listFoundModofVB.append(FoundNounInlist)
                if sentilist and len(sentilist) > 0:
                    for sentin in sentilist:
                        if sentin != 0:
                            if sentim_noun == 0:
                                sentim_noun = sentin
                            else:
                                if (sentim_noun > 0 and sentin < 0) or (
                                        sentim_noun < 0 and sentin > 0):  
                                    sentim_noun = np.sign(sentin) * np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  
                                else:  
                                    sentim_noun = np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  
                listFoundModofVB_sentim.append(sentim_noun)
            elif xin.dep_ == "prep" and xin.pos_ == "ADP" and xin.lemma_.lower() != t.lemma_.lower():
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                iterated_list_prep, iterated_list_prep_sentim, minw, maxw = PREP_token_IE_parsing(xin,singleNOUNs,singleCompoundedHITs,
                                                                                                  singleCompoundedHITs_toEXCLUDE,LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                                  VERBS_TO_KEEP, COMPUTE_OVERALL_SENTIMENT_SCORE, minw=minw,
                                                                                                  maxw=maxw,FoundVerb=FoundVerb,t=t,nountoskip=nountoskip)
                if iterated_list_prep and len(iterated_list_prep) > 0:
                    listFoundModofVB.extend(iterated_list_prep)
                    if iterated_list_prep_sentim and len(iterated_list_prep_sentim) > 0:
                        listFoundModofVB_sentim.extend(iterated_list_prep_sentim)
            if FoundNeg is None:
                for xinxin in lxin_n:
                    if (xinxin.dep_ == "neg"):
                        FoundNeg = "__not"
                        minw = min(minw, xinxin.i)
                        maxw = max(maxw, xinxin.i)
                for xinxin in rxin_n:
                    if (xinxin.dep_ == "neg"):
                        FoundNeg = "__not"
                        minw = min(minw, xinxin.i)
                        maxw = max(maxw, xinxin.i)
    sentim_vb = FeelIt(FoundVerb.lemma_.lower(), FoundVerb.pos_, FoundVerb)
    if not listFoundModofVB or len(listFoundModofVB) <= 0:
        if (FoundVerb.lemma_.lower() != "be" and FoundVerb.lemma_.lower() != "have"):
            FoundVerb_name = FoundVerb.lemma_.lower() + " ["+FoundVerb.pos_+", "+FoundVerb.tag_+" ("+str(sentim_vb)+ ")]" + FoundVerbAdverb
            listVerbs.append(FoundVerb_name)
            if FoundVerbAdverb_sentim != 0:
                if sentim_vb == 0:
                    sentim_vb = FoundVerbAdverb_sentim
                else:
                    if (sentim_vb > 0 and FoundVerbAdverb_sentim < 0) or (
                            sentim_vb < 0 and FoundVerbAdverb_sentim > 0): 
                        sentim_vb = np.sign(FoundVerbAdverb_sentim) * np.sign(sentim_vb) * (
                                abs(sentim_vb) + (1 - abs(sentim_vb)) * abs(
                            FoundVerbAdverb_sentim))  
                    else:  
                        sentim_vb = np.sign(sentim_vb) * (
                                abs(sentim_vb) + (1 - abs(sentim_vb)) * abs(
                            FoundVerbAdverb_sentim))  
            listVerbs_sentim = [sentim_vb]
    else:
        minw = min(minw, FoundVerb.i)
        maxw = max(maxw, FoundVerb.i)
        FoundVBInlist = "___" + FoundVerb.lemma_.lower() + " ["+FoundVerb.pos_+", "+FoundVerb.tag_+" ("+str(sentim_vb)+ ")]" + "___"
        for j in range(0, len(listFoundModofVB)):
            modin = listFoundModofVB[j]
            FoundVBInlist = FoundVBInlist + modin
            if j < len(listFoundModofVB) - 1 and (modin.endswith('__') == False):
                FoundVBInlist = FoundVBInlist + "__"
        for j in range(0, len(listFoundModofVB_sentim)):
            sentin = listFoundModofVB_sentim[j]
            if sentin != 0:
                if sentim_vb == 0:
                    sentim_vb = sentin
                else:
                    if (sentim_vb > 0 and sentin < 0) or (
                            sentim_vb < 0 and sentin > 0):  
                        sentim_vb = np.sign(sentin) * np.sign(sentim_vb) * (
                                abs(sentim_vb) + (1 - abs(sentim_vb)) * abs(
                            sentin)) 
                    else:  
                        sentim_vb = np.sign(sentim_vb) * (
                                abs(sentim_vb) + (1 - abs(sentim_vb)) * abs(
                            sentin))  
        listVerbs.append(FoundVBInlist)
        listVerbs_sentim.append(sentim_vb)
    listVerbs_app = []
    if FoundNeg == "__not" and len(listVerbs) > 0:
        for modin in listVerbs:
            listVerbs_app.append(modin + "__not")
        listVerbs = listVerbs_app
    return listVerbs, listVerbs_sentim, minw, maxw


def IsInterestingToken(t):
    ret = False
    if t.ent_type_ == "" or t.ent_type_ == 'ORG' or t.ent_type_ == 'GPE' or t.ent_type_ == 'PRODUCT' or t.ent_type_ == 'EVENT' or t.ent_type_ == 'LAW' or t.ent_type_ == 'MONEY' or t.ent_type_ == 'QUANTITY' or t.ent_type_ == 'LOC' or t.ent_type_ == 'NORP':  
        ret = True
    return ret


def NOUN_token_IE_parsing(t, singleNOUNs, singleCompoundedHITs, singleCompoundedHITs_toEXCLUDE,LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,minw, maxw, verbtoskip=None, nountoskip=None):
    to_give_back = []
    to_give_back_sentiment = []
    ll = [x for x in t.lefts]
    rr = [x for x in t.rights]
    CompoundsOfSingleHit = findCompoundedHITsOfTerm(singleCompoundedHITs, t.lemma_.lower())
    listVerbs = []
    listVerbs_sentim = []
    FoundVerb = None
    if t.head.pos_ == "VERB" and (not t.head is verbtoskip):
        FoundVerb = t.head
        minw = min(minw, FoundVerb.i)
        maxw = max(maxw, FoundVerb.i)
        lvin_n = [x for x in FoundVerb.lefts]
        rvin_n = [x for x in FoundVerb.rights]
        if lvin_n:
            for vin in lvin_n:
                lvin_inner = [x for x in vin.lefts]
                rvin_inner = [x for x in vin.rights]
                if (vin.dep_ == "xcomp" or vin.dep_ == "advcl") and vin.lemma_.lower() != t.lemma_.lower() and vin.lemma_.lower() != FoundVerb.lemma_.lower() and vin.pos_ == "VERB":
                    if (not verbtoskip) or (vin.lemma_.lower() != verbtoskip.lemma_.lower()):
                        minw = min(minw, vin.i)
                        maxw = max(maxw, vin.i)
                        list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(
                            vin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                            LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                            t, minw, maxw,nountoskip=nountoskip,previousverb=FoundVerb)
                        if list_verbs_app and len(list_verbs_app) > 0:
                            listVerbs.extend(list_verbs_app)
                            if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                                listVerbs_sentim.extend(list_verbs_sentim_app)
                elif (vin.dep_ == "acomp" or vin.dep_ == "oprd") and vin.lemma_.lower() != t.lemma_.lower() and \
                        vin.lemma_.lower() != FoundVerb.lemma_.lower():
                    if lvin_inner:
                        for vinvin in lvin_inner:
                            if (vinvin.dep_ == "xcomp" or vinvin.dep_ == "advcl") and vinvin.lemma_.lower() != t.lemma_.lower() and vinvin.lemma_.lower() != FoundVerb.lemma_.lower() and vinvin.pos_ == "VERB":
                                if (not verbtoskip) or (vinvin.lemma_.lower() != verbtoskip.lemma_.lower()):
                                    minw = min(minw, vinvin.i)
                                    maxw = max(maxw, vinvin.i)
                                    list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(
                                        vinvin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                        LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                        t,minw,maxw,nountoskip=nountoskip,previousverb=FoundVerb)
                                    if list_verbs_app and len(list_verbs_app) > 0:
                                        listVerbs.extend(list_verbs_app)
                                        if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                                            listVerbs_sentim.extend(list_verbs_sentim_app)
                    if rvin_inner:
                        for vinvin in rvin_inner:
                            if (vinvin.dep_ == "xcomp" or vinvin.dep_ == "advcl") and vinvin.lemma_.lower() != t.lemma_.lower() and vinvin.lemma_.lower() != FoundVerb.lemma_.lower() and vinvin.pos_ == "VERB":
                                if (not verbtoskip) or (vinvin.lemma_.lower() != verbtoskip.lemma_.lower()):
                                    minw = min(minw, vinvin.i)
                                    maxw = max(maxw, vinvin.i)
                                    list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(
                                        vinvin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                        LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                        t,minw,maxw,nountoskip=nountoskip,previousverb=FoundVerb)
                                    if list_verbs_app and len(list_verbs_app) > 0:
                                        listVerbs.extend(list_verbs_app)
                                        if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                                            listVerbs_sentim.extend(list_verbs_sentim_app)
        if rvin_n:
            for vin in rvin_n:
                lvin_inner = [x for x in vin.lefts]
                rvin_inner = [x for x in vin.rights]
                if (vin.dep_ == "xcomp" or vin.dep_ == "advcl") and vin.lemma_.lower() != t.lemma_.lower() and vin.lemma_.lower() != FoundVerb.lemma_.lower() and vin.pos_ == "VERB":
                    if (not verbtoskip) or (vin.lemma_.lower() != verbtoskip.lemma_.lower()):
                        minw = min(minw, vin.i)
                        maxw = max(maxw, vin.i)
                        list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(
                            vin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                            LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                            t, minw, maxw,nountoskip=nountoskip,previousverb=FoundVerb)
                        if list_verbs_app and len(list_verbs_app) > 0:
                            listVerbs.extend(list_verbs_app)
                            if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                                listVerbs_sentim.extend(list_verbs_sentim_app)
                elif (vin.dep_ == "acomp" or vin.dep_ == "oprd") and vin.lemma_.lower() != t.lemma_.lower() and vin.lemma_.lower() != FoundVerb.lemma_.lower():
                    if lvin_inner:
                        for vinvin in lvin_inner:
                            if (
                                    vinvin.dep_ == "xcomp" or vinvin.dep_ == "advcl") and vinvin.lemma_.lower() != t.lemma_.lower() and vinvin.lemma_.lower() != FoundVerb.lemma_.lower() and vinvin.pos_ == "VERB": 
                                if (not verbtoskip) or (vinvin.lemma_.lower() != verbtoskip.lemma_.lower()):
                                    minw = min(minw, vinvin.i)
                                    maxw = max(maxw, vinvin.i)
                                    list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(vinvin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                                              LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                                              t,minw,maxw,nountoskip=nountoskip,previousverb=FoundVerb)
                                    if list_verbs_app and len(list_verbs_app) > 0:
                                        listVerbs.extend(list_verbs_app)
                                        if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                                            listVerbs_sentim.extend(list_verbs_sentim_app)
                    if rvin_inner:
                        for vinvin in rvin_inner:
                            if (
                                    vinvin.dep_ == "xcomp" or vinvin.dep_ == "advcl") and vinvin.lemma_.lower() != t.lemma_.lower() and vinvin.lemma_.lower() != FoundVerb.lemma_.lower() and vinvin.pos_ == "VERB": 
                                if (not verbtoskip) or (vinvin.lemma_.lower() != verbtoskip.lemma_.lower()):
                                    minw = min(minw, vinvin.i)
                                    maxw = max(maxw, vinvin.i)
                                    list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(vinvin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                                              LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                                              t,minw,maxw,nountoskip=nountoskip,previousverb=FoundVerb)
                                    if list_verbs_app and len(list_verbs_app) > 0:
                                        listVerbs.extend(list_verbs_app)
                                        if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                                            listVerbs_sentim.extend(list_verbs_sentim_app)
        if (not verbtoskip) or (FoundVerb.lemma_.lower() != verbtoskip.lemma_.lower()):
            minw = min(minw, FoundVerb.i)
            maxw = max(maxw, FoundVerb.i)
            list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(FoundVerb, singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                      LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                      t, minw=minw, maxw=maxw,nountoskip=nountoskip)
            if list_verbs_app and len(list_verbs_app) > 0:
                listVerbs.extend(list_verbs_app)
                if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                    listVerbs_sentim.extend(list_verbs_sentim_app)
    # ------------------------------------------------------------------------------------------------
    listAMODs = []
    listAMODs_sentim = []
    FoundAMOD = None
    FoundNeg_left = None
    if ll:
        for xin in ll:
            lxin_n = [x for x in xin.lefts]
            rxin_n = [x for x in xin.rights]
            if (xin.dep_ == "neg"):
                FoundNeg_left = "__not"
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
            elif (xin.dep_ == "amod" and (
                    (xin.pos_ == "ADJ" and (xin.tag_ == "JJR" or xin.tag_ == "JJS" or xin.tag_ == "JJ")) or (
                    xin.pos_ == "VERB")) and xin.lemma_.lower() != t.lemma_.lower()):  
                if (xin.lemma_.lower() in CompoundsOfSingleHit):
                    continue
                FoundAMOD = xin
                FoundAMOD_sentiment = FeelIt(FoundAMOD.lemma_.lower(), FoundAMOD.pos_, FoundAMOD)
                listAMODs_sentim.append(FoundAMOD_sentiment)
                FoundAMOD_name = FoundAMOD.lemma_.lower() + " ["+FoundAMOD.pos_+", "+FoundAMOD.tag_+" ("+str(FoundAMOD_sentiment)+ ")]"
                listAMODs.append(FoundAMOD_name)
                minw = min(minw, FoundAMOD.i)
                maxw = max(maxw, FoundAMOD.i)
            elif xin.dep_ == "acl" and xin.lemma_.lower() != t.lemma_.lower():
                if (
                        xin.pos_ == "VERB"):  
                    minw = min(minw, xin.i)
                    maxw = max(maxw, xin.i)
                    iterated_list_VERB, iterated_list_VERB_sentiment, minw, maxw = VERB_token_IE_parsing(xin, singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                                         LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                                         t, minw,maxw,nountoskip=nountoskip,previousverb=FoundVerb)
                    if iterated_list_VERB and len(iterated_list_VERB) > 0:
                        listAMODs.extend(iterated_list_VERB)
                        if iterated_list_VERB_sentiment and len(iterated_list_VERB_sentiment) > 0:
                            listAMODs_sentim.extend(iterated_list_VERB_sentiment)
            elif xin.dep_ == "prep" and xin.pos_ == "ADP" and xin.lemma_.lower() != t.lemma_.lower() and (
                    nountoskip is None):  
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                iterated_list_prep, iterated_list_prep_sentim, minw, maxw = PREP_token_IE_parsing(xin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                                  minw=minw,maxw=maxw,FoundVerb=FoundVerb,t=t,nountoskip=nountoskip)
                if iterated_list_prep and len(iterated_list_prep) > 0:
                    listAMODs.extend(iterated_list_prep)
                    if iterated_list_prep_sentim and len(iterated_list_prep_sentim) > 0:
                        listAMODs_sentim.extend(iterated_list_prep_sentim)
    FoundNeg_right = None
    if rr:
        for xin in rr:
            lxin_n = [x for x in xin.lefts]
            rxin_n = [x for x in xin.rights]
            if (xin.dep_ == "neg"):
                FoundNeg_right = "__not"
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
            elif (xin.dep_ == "amod" and (
                    (xin.pos_ == "ADJ" and (xin.tag_ == "JJR" or xin.tag_ == "JJS" or xin.tag_ == "JJ")) or (
                    xin.pos_ == "VERB")) and xin.lemma_.lower() != t.lemma_.lower()):  
                if (xin.lemma_.lower() in CompoundsOfSingleHit):
                    continue
                FoundAMOD = xin
                FoundAMOD_sentiment = FeelIt(FoundAMOD.lemma_.lower(), FoundAMOD.pos_, FoundAMOD)
                listAMODs_sentim.append(FoundAMOD_sentiment)
                FoundAMOD_name = FoundAMOD.lemma_.lower() + " ["+FoundAMOD.pos_+", "+FoundAMOD.tag_+" ("+str(FoundAMOD_sentiment)+ ")]"
                listAMODs.append(FoundAMOD_name)
                minw = min(minw, FoundAMOD.i)
                maxw = max(maxw, FoundAMOD.i)
            elif xin.dep_ == "acl" and xin.lemma_.lower() != t.lemma_.lower():
                if xin.pos_ == "VERB":
                    minw = min(minw, xin.i)
                    maxw = max(maxw, xin.i)
                    iterated_list_VERB, iterated_list_VERB_sentiment, minw, maxw  = VERB_token_IE_parsing(xin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                                          LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                                          t, minw,maxw,nountoskip=nountoskip,previousverb=FoundVerb)
                    if iterated_list_VERB and len(iterated_list_VERB) > 0:
                        listAMODs.extend(iterated_list_VERB)
                        if iterated_list_VERB_sentiment and len(iterated_list_VERB_sentiment) > 0:
                            listAMODs_sentim.extend(iterated_list_VERB_sentiment)
            elif xin.dep_ == "prep" and xin.pos_ == "ADP" and xin.lemma_.lower() != t.lemma_.lower() and (
                    nountoskip is None):  
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                iterated_list_prep, iterated_list_prep_sentim, minw, maxw = PREP_token_IE_parsing(xin,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                                  minw=minw,maxw=maxw,FoundVerb=FoundVerb,t=t,nountoskip=nountoskip)
                if iterated_list_prep and len(iterated_list_prep) > 0:
                    listAMODs.extend(iterated_list_prep)
                    if iterated_list_prep_sentim and len(iterated_list_prep_sentim) > 0:
                        listAMODs_sentim.extend(iterated_list_prep_sentim)
    listAMODs_app = []
    if (FoundNeg_left == "__not" or FoundNeg_right == "__not") and len(listAMODs) > 0:
        for modin in listAMODs:
            listAMODs_app.append(modin + "__not")
        listAMODs = listAMODs_app
    if len(listAMODs) > 0:
        to_give_back.extend(listAMODs)
    if len(listVerbs) > 0:
        to_give_back.extend(listVerbs)
    if len(listAMODs_sentim) > 0:
        to_give_back_sentiment.extend(listAMODs_sentim)
    if len(listVerbs_sentim) > 0:
        to_give_back_sentiment.extend(listVerbs_sentim)
    return to_give_back, to_give_back_sentiment, minw, maxw


def determine_tense_input(tagged, posextractedn):
    for ww in tagged:
        if ww.pos_ == "VERB" and ww.dep_ == "aux" and (
                ww.tag_ == "VBP" or ww.tag_ == "VBZ") and ww.head.lower_ == "going":
            lll = [x for x in ww.head.rights]
            for xxx in lll:
                if xxx.pos_ == "VERB" and xxx.tag_ == "VB":
                    ww.tag_ = "MD"
    tense = {}
    future_words = [word for word in tagged if word.tag_ == "MD"]
    present_words = [word for word in tagged if word.tag_ in ["VBP", "VBZ", "VBG"]]
    pass_words = [word for word in tagged if word.tag_ in ["VBD", "VBN"]]
    inf_words = [word for word in tagged if word.tag_ in ["VB"]]  
    valfuture = 0
    for word in future_words:
        valfuture = valfuture + 1 / abs(posextractedn - word.i)
    valpresent = 0
    for word in present_words:
        valpresent = valpresent + 1 / abs(posextractedn - word.i)
    valpass = 0
    for word in pass_words:
        valpass = valpass + 1 / abs(posextractedn - word.i)
    valinf = 0
    for word in inf_words:
        valinf = valinf + 1 / abs(posextractedn - word.i)
    tense["future"] = valfuture
    tense["present"] = valpresent
    tense["past"] = valpass
    return (tense)


def keep_token_IE(t, singleNOUNs, singleCompoundedHITs, singleCompoundedHITs_toEXCLUDE, most_frequent_loc_DOC, LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_SENTIMENT_SCORE, MOST_FREQ_LOC_HEURISTIC):
    to_give_back = []
    sentiment_to_give_back = []
    spantogiveback = []
    textsentencetogiveback = []
    tensetogiveback = []
    if t.is_alpha and not (t.is_space or t.is_punct or t.is_stop or t.like_num) and t.pos_ == "NOUN":
        CompoundsOfSingleHit = findCompoundedHITsOfTerm(singleCompoundedHITs, t.lemma_.lower())
        if (t.lemma_.lower() in singleNOUNs) or (CompoundsOfSingleHit):
            FoundCompound = None
            ll = [x for x in t.lefts]
            if not FoundCompound:
                if ll:
                    for xin in ll:
                        if ((
                                xin.lemma_.lower() in CompoundsOfSingleHit)) and xin.lemma_.lower() != t.lemma_.lower():
                            FoundCompound = xin
                            break
            rr = [x for x in t.rights]
            if not FoundCompound:
                if rr:
                    for xin in rr:
                        if ((
                                xin.lemma_.lower() in CompoundsOfSingleHit)) and xin.lemma_.lower() != t.lemma_.lower():
                            FoundCompound = xin
                            break
            if FoundCompound or t.lemma_.lower() in singleNOUNs:
                if singleCompoundedHITs_toEXCLUDE:
                    CompoundsOfSingleHitToExclude = findCompoundedHITsOfTerm(singleCompoundedHITs_toEXCLUDE,
                                                                             t.lemma_.lower())
                    if CompoundsOfSingleHitToExclude:
                        if ll:
                            for xin in ll:
                                if xin.lemma_.lower() in CompoundsOfSingleHitToExclude:
                                    return to_give_back, sentiment_to_give_back, spantogiveback, \
                                           textsentencetogiveback,tensetogiveback  
                        if rr:
                            for xin in rr:
                                if xin.lemma_.lower() in CompoundsOfSingleHitToExclude:
                                    return to_give_back, sentiment_to_give_back, spantogiveback, \
                                           textsentencetogiveback,tensetogiveback  
                    #
                if MOST_FREQ_LOC_HEURISTIC is True:
                    most_frequent_loc_SENTENCE = determine_location_heuristic(t.sent.ents, t.i, t,
                                                                              LOCATION_SYNONYMS_FOR_HEURISTIC)
                    if most_frequent_loc_DOC == LOCATION_SYNONYMS_FOR_HEURISTIC[0].lower():
                        if (most_frequent_loc_SENTENCE == LOCATION_SYNONYMS_FOR_HEURISTIC[0].lower() or most_frequent_loc_SENTENCE == "") == False:
                            return to_give_back, sentiment_to_give_back, spantogiveback, \
                                   textsentencetogiveback, tensetogiveback 
                    else:  
                        if most_frequent_loc_SENTENCE != LOCATION_SYNONYMS_FOR_HEURISTIC[0].lower():
                            return to_give_back, sentiment_to_give_back, spantogiveback, \
                                   textsentencetogiveback, tensetogiveback  
                if FoundCompound:
                    minw = min(FoundCompound.i, t.i)
                    maxw = max(FoundCompound.i, t.i)
                else:
                    minw = t.i
                    maxw = t.i
                if COMPUTE_OVERALL_SENTIMENT_SCORE is True:
                    OSpolarity = FeelIt_OverallSentiment(t)
                    toveralltestsentece__ = t.sent.text.replace(" ", "__")
                    to_give_back.append(toveralltestsentece__)
                    sentiment_to_give_back.append(OSpolarity)
                    minw = t.sent.start
                    maxw = t.sent.end - 1
                else:
                    to_give_back, sentiment_to_give_back, minw, maxw = NOUN_token_IE_parsing(t,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,
                                                                                             LOCATION_SYNONYMS_FOR_HEURISTIC,VERBS_TO_KEEP,COMPUTE_OVERALL_SENTIMENT_SCORE,
                                                                                             minw=minw,maxw=maxw)
                tryl = True
                while tryl == True:
                    if t.doc[minw].pos_ == "VERB":
                        tryl = False
                        for xis in t.doc[minw].lefts:
                            if (xis.dep_ == "aux" and xis.pos_ == "VERB"):
                                minw = xis.i
                                tryl = True
                    else:
                        tryl = False
                tryl = True
                while tryl == True:
                    if t.doc[maxw].pos_ == "VERB":
                        tryl = False
                        for xis in t.doc[maxw].rights:
                            if (xis.dep_ == "aux" and xis.pos_ == "VERB"):
                                maxw = xis.i
                                tryl = True
                    else:
                        tryl = False
                spansentence = t.doc[minw:(maxw + 1)]
                tensedict = determine_tense_input(spansentence, t.i)
                tense = "NaN"
                tupletense = max(tensedict.items(), key=operator.itemgetter(1))  # [0]
                if tupletense[1] > 0:
                    tense = tupletense[0]
                tensetogiveback = [str(tense)]
                if (tense in VERBS_TO_KEEP) == False:
                    return [], [], [], [], []
                if sentiment_to_give_back and len(sentiment_to_give_back) > 0:
                    sentim_noun = FeelIt(t.lemma_.lower(), t.pos_, t)
                if to_give_back and len(to_give_back) > 0:
                    if len(to_give_back) == len(sentiment_to_give_back):
                        listNoun_app = []
                        listSentim_app = []
                        if FoundCompound:
                            FoundNounInlist_ALLTOGETHER = "---" + FoundCompound.lemma_.lower() + " " + t.lemma_.lower() + "---"
                        else:
                            FoundNounInlist_ALLTOGETHER = "---" + t.lemma_.lower() + "---"
                        numberofvaluesent = 0
                        sumofvaluessent = 0
                        for ii in range(len(to_give_back)):
                            modin = to_give_back[ii]
                            sentin = sentiment_to_give_back[ii]
                            if FoundCompound:
                                FoundNounInlist = "---" + FoundCompound.lemma_.lower() + " " + t.lemma_.lower() + "---" + modin
                            else:
                                FoundNounInlist = "---" + t.lemma_.lower() + "---" + modin
                            listNoun_app.append(FoundNounInlist)
                            FoundNounInlist_ALLTOGETHER = FoundNounInlist_ALLTOGETHER + "+++" + modin
                            if sentin != 0 and sentim_noun != 0:
                                sentin = np.sign(sentin) * np.sign(sentim_noun) * abs(sentin)  
                            if "__not" in FoundNounInlist:
                                sentin = (-1) * sentin
                            listSentim_app.append(sentin)
                            if sentin != 0:
                                numberofvaluesent = numberofvaluesent + 1
                                sumofvaluessent = sumofvaluessent + sentin
                        listNoun_app2 = []
                        listSentim_app2 = []
                        listNoun_app2.append(FoundNounInlist_ALLTOGETHER)
                        if numberofvaluesent > 0:
                            avgsent_ALLTOGETHER = sumofvaluessent / numberofvaluesent
                        else:
                            avgsent_ALLTOGETHER = 0
                        listSentim_app2.append(avgsent_ALLTOGETHER)
                        to_give_back = listNoun_app2
                        sentiment_to_give_back = listSentim_app2
                        spantogiveback = [spansentence.text]
                        textsentencetogiveback = [t.sent.text]
    return to_give_back, sentiment_to_give_back, spantogiveback, textsentencetogiveback, tensetogiveback


def Most_Common(lista):
    country = ""
    if lista:
        data = Counter(lista)
        ordered_c = data.most_common()
        country = ordered_c[0][0]
        max_freq = ordered_c[0][1]
        for j in range(0, len(ordered_c)):
            if ordered_c[j][1] < max_freq:
                break
    return country


def findCompoundedHITsOfTerm(vector, term):
    term = str(term)
    outArray = []
    for x in vector:
        if term.lower() in x.lower():
            compoundMinusTerm = x.lower().replace(term.lower(), "").strip()
            outArray.append(compoundMinusTerm)
    return outArray


def determine_location_heuristic(doc_entities, posextractedn, t,LOCATION_SYNONYMS_FOR_HEURISTIC):
    most_probable_loc = ""
    tagged = []
    for loc in doc_entities:
        if loc.label_ == "GPE" or loc.label_ == "NORP" or loc.label_ == "LOC" or loc.label_ == "ORG":
            tagged.append(loc)
    if len(tagged) > 0:
        unique_loc_labels = []
        unique_loc_values = []
        for loc in tagged:
            x = loc.lemma_.lower()
            if (x in LOCATION_SYNONYMS_FOR_HEURISTIC) or (removearticles(x) in LOCATION_SYNONYMS_FOR_HEURISTIC):
                x = LOCATION_SYNONYMS_FOR_HEURISTIC[0].lower()
            if x not in unique_loc_labels:
                unique_loc_labels.append(x)
                valword = 0
                for word in tagged:
                    y = word.lemma_.lower()
                    if ((y in LOCATION_SYNONYMS_FOR_HEURISTIC) or (
                            removearticles(y) in LOCATION_SYNONYMS_FOR_HEURISTIC)):
                        y = LOCATION_SYNONYMS_FOR_HEURISTIC[0].lower()
                    if y == x:
                        avgpostne = word.start + int((word.end - word.start) / 2)
                        dividendum = abs(posextractedn - avgpostne)
                        if dividendum == 0:
                            dividendum = 1
                        valword = valword + 1 / dividendum
                unique_loc_values.append(valword)
        maxvalue = max(unique_loc_values)
        indices_max = [i for i, x in enumerate(unique_loc_values) if x == maxvalue]
        most_probable_loc = unique_loc_labels[indices_max[0]] 
        for wwind in indices_max:  
            ww = unique_loc_labels[wwind]
            if ww == LOCATION_SYNONYMS_FOR_HEURISTIC[0].lower():
                most_probable_loc = ww
    return most_probable_loc


def removearticles(text):
    removed = re.sub('\s+(a|an|and|the)(\s+)', ' ', " " + text + " ")
    removed = re.sub('  +', ' ', removed)
    removed = removed.strip()
    return removed


def lemmatize_doc_IE_Sentiment(doc,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_SENTIMENT_SCORE,MOST_FREQ_LOC_HEURISTIC):
    vect = []
    vect_sentiment = []
    vect_spans = []
    vect_text = []
    vect_tense = []
    if MOST_FREQ_LOC_HEURISTIC is True:   
        locations = [loc.lemma_.lower() for loc in doc.ents if
                     (loc.label_ == "GPE" or loc.label_ == "NORP" or loc.label_ == "LOC" or loc.label_ == "ORG" )]
        locations = [LOCATION_SYNONYMS_FOR_HEURISTIC[0].lower() if ((x in LOCATION_SYNONYMS_FOR_HEURISTIC) or (
                removearticles(x) in LOCATION_SYNONYMS_FOR_HEURISTIC)) else x for x in locations]
        most_frequent_loc = Most_Common(locations)
    else:
        most_frequent_loc = None
    sentencealreadyseen = ""
    for t in doc:                                 
        vec_for_term, vec_for_sent, spansse, texttse, tensesse = keep_token_IE(t,singleNOUNs,singleCompoundedHITs,singleCompoundedHITs_toEXCLUDE,most_frequent_loc, LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_SENTIMENT_SCORE, MOST_FREQ_LOC_HEURISTIC)
        if vec_for_term:
            if len(vec_for_term) > 0:
                if COMPUTE_OVERALL_SENTIMENT_SCORE == True:
                    thissentence = str(t.sent.text)
                    if (thissentence == sentencealreadyseen):
                        continue
                    else:
                        sentencealreadyseen = str(t.sent.text)
                        vect.extend(vec_for_term)
                        vect_sentiment.extend(vec_for_sent)
                        vect_spans.append(spansse)
                        vect_text.append(texttse)
                        vect_tense.append(tensesse)
                else:
                    vect.extend(vec_for_term)
                    vect_sentiment.extend(vec_for_sent)
                    vect_spans.append(spansse)
                    vect_text.append(texttse)
                    vect_tense.append(tensesse)
    return vect, vect_sentiment, vect_spans, vect_text, vect_tense


def CheckLeapYear(year):
    isleap = False
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                isleap = True
            else:
                isleap = False
        else:
            isleap = True
    else:
        isleap = False
    return isleap


def get_sentiment(text, include, exclude=None, location=None, tense=['past', 'present', 'future', 'NaN'], oss=False):
    # text = ['Today is a beautiful day', 'The economy is slowing down and it is a rainy day']
    # include = ['day', 'economy']
    # exclude=None 
    # location=None
    # tense=['past', 'present', 'future', 'NaN']
    # oss=False
    toINCLUDE = include
    singleCompoundedHITs_toEXCLUDE = exclude
    LOCATION_SYNONYMS_FOR_HEURISTIC = location
    VERBS_TO_KEEP = tense
    COMPUTE_OVERALL_SENTIMENT_SCORE = oss
    for i in range(len(text)):
        text[i] = re.sub("\n \\n", " ", str(text[i]))
        
    if LOCATION_SYNONYMS_FOR_HEURISTIC and len(LOCATION_SYNONYMS_FOR_HEURISTIC) > 0:
        MOST_FREQ_LOC_HEURISTIC = True
    else:
        MOST_FREQ_LOC_HEURISTIC = False
    singleNOUNs = []
    singleCompoundedHITs = []
    for ii in toINCLUDE:
        if " " in ii:
            singleCompoundedHITs.append(ii)
        else:
            singleNOUNs.append(ii)
            
    currentDT = datetime.now()
    spacy_model_name_EN = 'en_core_web_lg'
    # from timeit import default_timer as timer
    # start = timer()
    # print("spaCy is loading the en_core_web_lg model ...")
    nlp_EN = spacy.load(spacy_model_name_EN) ## this operation takes approximately 10seconds
    # print(timer()-start) ## elapsed time in seconds
    LA_target = 'en'
    docs_lemma = []
    docs_lemma_sentiment = []
    docsspans = []
    docstexttt = []
    docstense = []
    DF_ExtractionsSummary = []

    for j in range(len(text)):
        nlp_COUNTRYdoc = nlp_EN(text[j])
        lemmatized_doc, lemmatized_doc_sent, spanss, texttt, tensesss = lemmatize_doc_IE_Sentiment(
            nlp_COUNTRYdoc, singleNOUNs, singleCompoundedHITs, singleCompoundedHITs_toEXCLUDE,
            LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_SENTIMENT_SCORE, MOST_FREQ_LOC_HEURISTIC)
        docs_lemma.append(lemmatized_doc)
        docs_lemma_sentiment.append(lemmatized_doc_sent)
        docsspans.append(spanss)
        docstexttt.append(texttt)
        docstense.append(tensesss)
        for i in range(len(docstexttt[j])):
            includedNOUN = []
            check = (singleNOUNs + singleCompoundedHITs)
            for k in check:
                if k in str(docsspans[j][i]).lower():
                    includedNOUN.append(k)
            DF_ExtractionsSummary.append([j, docstexttt[j][i], docsspans[j][i], docs_lemma[j][i],
                                          docs_lemma_sentiment[j][i], docstense[j][i], includedNOUN])

    DF_ExtractionsSummary = pd.DataFrame(DF_ExtractionsSummary, columns=['Doc_id', 'Text', 'SpannedText', 'Chunk',
                                                                         'Sentiment', 'Tense', 'Include'])
    return DF_ExtractionsSummary
