#Syntax and Style

from itertools import count
from itertools import count
import os
import glob
from xml.parsers.expat import model 
import pandas as pd
import nltk
import string
import re
from nltk.corpus import cmudict
import spacy
import pickle
import math


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
#nlp = spacy.load("en_core_web_sm") #stored outside of the function to prevent it being loaded each run 

def parse(txt_files): #processing text using spaCy tokenizing and parsing + added to dataframe
    novels = []
    for txt_file in txt_files:
       with open(txt_file, 'r') as file:
            content = file.read()
            filename = os.path.basename(txt_file).split('-')
            title = filename[0]
            doc = nlp(content)
            novels.append({'Text': content, 'Title': filename[0], 'Author': filename[1], 'Year of Publication': filename[2], 'Doc': doc})
    spacy_dataframe = pd.DataFrame(novels)
    with open('dataframe.pkl', 'wb') as file:
       pickle.dump(spacy_dataframe, file)
    return spacy_dataframe

spacy_dataframe = parse(txt_files)
print(spacy_dataframe)

###

#1.e:

#load the dataframe from the pickle file
with open('dataframe.pkl', 'rb') as file:
    spacy_dataframe = pickle.load(file)

#for loop for top 10 syntactic subjects per novel.
top_10_subjects = {}
for index, row in spacy_dataframe.iterrows():
    doc = row['Doc']
    subjects = [token.text for token in doc if token.dep_ == 'nsubj']
    subject_freq = nltk.FreqDist(subjects)
    top_10_subjects[row['Title']] = subject_freq.most_common(10)

#additional for loop to make presentation of output easier to differentiate between novels
for title, subjects in top_10_subjects.items():
    print(f"Top 10 syntactic subjects in {title}:")
    print(subjects)

#####

#for loop to output top verbs associated with He (ordered by PMI)
top_verbs_he = {}
for index,row in spacy_dataframe.iterrows():
    verb_counter = {}
    he_counter = 0
    pmi_data_he = []
    for token in row['Doc']:
        if token.text.lower() == 'he' and token.dep_ in ['nsubj']:
            head = token.head
            he_counter += 1
            if head.pos_ == "VERB":
                verb_counter[head.text] = verb_counter.get(head.text, 0) + 1
    for verb, count in verb_counter.items():
        if count >= 5:
            p_w1_w2 = (count + he_counter) / len(row['Text'].split())
            p_w1 = count / len(row['Text'].split())
            p_w2 = he_counter / len(row['Text'].split())
            pmi = round(math.log2(p_w1_w2 / (p_w1 * p_w2)),2)
            pmi_data_he.append((verb, pmi)) 
    top_verbs_he[row['Title']] = sorted(pmi_data_he)

#additional for loop to make presentation of output easier to differentiate between novels   
for title, verbs in top_verbs_he.items():
    print(f"Top verbs associated with 'he' in {title}:")
    print(verbs)


####

#for loop to output top verbs associated with She (ordered by PMI)

top_verbs_she = {}
for index,row in spacy_dataframe.iterrows():
    verb_counter = {}
    she_counter = 0
    pmi_data_she = []
    for token in row['Doc']:
        if token.text.lower() == 'she' and token.dep_ in ['nsubj']:
            head = token.head
            she_counter += 1
            if head.pos_ == 'VERB':
                verb_counter[head.text] = verb_counter.get(head.text, 0) + 1
    for verb, count in verb_counter.items():
        if count >= 5:
            p_w1_w2 = (count + she_counter) / len(row['Text'].split())
            p_w1 = count / len(row['Text'].split())
            p_w2 = she_counter / len(row['Text'].split())
            pmi = round(math.log2(p_w1_w2 / (p_w1 * p_w2)),2)
            pmi_data_she.append((verb, pmi)) 
    top_verbs_she[row['Title']] = sorted(pmi_data_she)

#additional for loop to make presentation of output easier to differentiate between novels

for title, verbs in top_verbs_she.items():
    print(f"Top verbs associated with 'she' in {title}:")
    print(verbs)









