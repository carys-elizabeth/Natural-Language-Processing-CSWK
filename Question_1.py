#Syntax and Style

import os
import glob 
import pandas as pd
import nltk
import string
import re
from nltk.corpus import cmudict


#1.a:

directory_path = "/Users/caryswilliams/Documents/Masters Degree Folder/Coursework Pack NLP/texts"

txt_files = glob.glob(os.path.join(directory_path, "**", "*.txt"), recursive=True)

def read_novels(txt_files): #dataframe with important information about the novels, ordered by year of publication
    novels = []
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            content = file.read()
            filename = os.path.basename(txt_file).split('-')
            novels.append({'Text': content, 'Title': filename[0], 'Author': filename[1], 'Year of Publication': filename[2]})
    novels_dataframe = pd.DataFrame(novels)
    
    return novels_dataframe.sort_values(by='Year of Publication', ignore_index=True)

#novels_dataframe = read_novels(txt_files)
#print(novels_dataframe)

#1.b:

def nltk_ttr(txt_files): #dictionary, mapping title to type text ratio, no punctuation involved and ignoring case 
    dict_ttr = {}
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            content = file.read()
            filename = os.path.basename(txt_file).split('-')
            title = filename[0]
            tokens = nltk.word_tokenize(content.lower())
            tokens = [x for x in tokens if x.isalnum()]
            if not tokens:
                return 0
            types = set(tokens)
            ttr = (len(types)) / len(tokens) #ttr - number of unique word types div by total number of word tokens
            dict_ttr[title] = ttr
    return dict_ttr

#1.c:

cmu_dict = cmudict.dict()

def count_syllables(word): #CMU for syllables
    if word.lower() not in cmu_dict:
        return 0
    syllable_counts = (len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word.lower()])
    return next(syllable_counts, 0)

def flesch_kincaid(txt_files): #dictionary, map title to fk reading ease score with CMU for estimating syllables
    dict_fk = {}
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            content = file.read()
            filename = os.path.basename(txt_file).split('-')
            title = filename[0]
            tokens = nltk.word_tokenize(content.lower())
            tokens = [x for x in tokens if x.isalnum()]
            if not tokens:
                dict_fk[title] = 0
                continue
            syllables = sum(count_syllables(word) for word in tokens)
            sentences = nltk.sent_tokenize(content)
            dict_fk[title] = 0.39 * (len(tokens) / len(sentences)) + 11.8 * (syllables / len(tokens)) - 15.59
    return dict_fk

#1.d:







