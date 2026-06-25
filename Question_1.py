#Syntax and Style

import os
import glob
import pandas as pd
import nltk
from nltk.corpus import cmudict
from pathlib import Path
import spacy
import pickle
import math

#Question 1.a: all functions, corresponding to parts of question, will run upon calling in the terminal 

directory_path = "/Users/caryswilliams/Documents/Masters Degree Folder/Coursework Pack NLP/texts" #input own directory path here 
txt_files = glob.glob(os.path.join(directory_path, "**", "*.txt"), recursive=True)

def read_novels(txt_files): #dataframe with information about the novels, ordered by year of publication
    novels = []
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            content = file.read()
            filename = os.path.basename(txt_file).split('-')
            novels.append({'Text': content, 'Title': filename[0], 'Author': filename[1], 'Year of Publication': filename[2]})
    novels_dataframe = pd.DataFrame(novels)
    
    return novels_dataframe.sort_values(by='Year of Publication', ignore_index=True)


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
            ttr = (len(types)) / len(tokens) #ttr calculation
            dict_ttr[title] = ttr
    return dict_ttr 

#1.c:

cmu_dict = cmudict.dict()
def count_syllables(word): #function for syllable count
    if word.lower() not in cmu_dict:
        return 0
    syllable_counts = (len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word.lower()])
    return next(syllable_counts, 0)

def flesch_kincaid(txt_files): #dictionary, map title to fk reading ease score 
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
nlp = spacy.load("en_core_web_sm") #stored outside of the function to prevent it being loaded each run 
def parse(txt_files): #processing text using spaCy tokenizing and parsing + added to dataframe, saved to a pickle file
    novels = []
    for txt_file in txt_files:
       with open(txt_file, 'r') as file:
            content = file.read()
            filename = os.path.basename(txt_file).split('-')
            doc = nlp(content)
            novels.append({'Text': content, 'Title': filename[0], 'Author': filename[1], 'Year of Publication': filename[2], 'Doc': doc})
    spacy_dataframe = pd.DataFrame(novels)
    output_file = Path("dataframe.pkl")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, 'wb+') as file:
       pickle.dump(spacy_dataframe, file)
    return spacy_dataframe

#1.e:

#3x functions, each holding for loop for top 10 syntactic subjects per novel.
#with this I have understood "subject" to mean the subject of the clause, therefore counting the highest number of "nsubj"
def top_10_subjects(spacy_dataframe): #function to output the top 10 syntactic subjects in each novel
    top_10_subjects = {}
    for index, row in spacy_dataframe.iterrows():
        doc = row['Doc']
        subjects = [token.text for token in doc if token.dep_ == 'nsubj']
        subject_freq = nltk.FreqDist(subjects)
        top_10_subjects[row['Title']] = subject_freq.most_common(10)

    for title, subjects in top_10_subjects.items():
        print(f"Top 10 syntactic subjects in {title}:")
        print(subjects)

#####

def top_verbs_he(spacy_dataframe): #function to output top verbs associated with He (ordered by PMI)
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
   
    for title, verbs in top_verbs_he.items():
        print(f"Top verbs associated with 'he' in {title}:")
        print(verbs)

####

def top_verbs_she(spacy_dataframe) : #function to output top verbs associated with She (ordered by PMI)
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
    for title, verbs in top_verbs_she.items():
        print(f"Top verbs associated with 'she' in {title}:")
        print(verbs)

if __name__ == "__main__":
    print("Initial Novels Dataframe:\n")
    novels_dataframe = read_novels(txt_files)
    print(novels_dataframe)
    print("\nDictionary of Type Token Ratio:\n\n")
    nltk_ttr_calc = nltk_ttr(txt_files)
    print(nltk_ttr_calc)
    print("\nDictionary of Flesch Kincaid Scores:\n\n")
    flesch_kincaid_calc = flesch_kincaid(txt_files)
    print(flesch_kincaid_calc)
    print("\nSpacy Dataframe:\n\n")
    parse_spacy_dataframe = parse(txt_files)
    print(parse_spacy_dataframe)
    with open('dataframe.pkl', 'rb') as file: #loading the dataframe from the pickle file
        spacy_dataframe = pickle.load(file)
    print("\nTop 10 Subtactic Subjects in Novels(Ordered by PMI):\n")
    top_10_subj = top_10_subjects(spacy_dataframe)
    print(top_10_subj)
    print("\nTop Verbs associated with He in Novels(Ordered by PMI):\n")
    top_he = top_verbs_he(spacy_dataframe)
    print(top_he)
    print("\nTop Verbs associated with She in Novels(Ordered by PMI):\n")
    top_she = top_verbs_she(spacy_dataframe)
    print(top_she)



#AI acknowledgement:
#Within this Question, the built in CoPilot in VScode was used for minor troubleshooting of errors:
#1 Where en_core_web_sm was not installing correctly, due to a missing install, 
#2 Due to an issue with variable names being repeated (cmudict)
#3 Where the data being processed in read_novels(txt_files) was not being outputted correctly.
#No code was directly generated and used in Question 1, all suggestions by CoPilot were reviewed in the CoPilot window but adapted and implemented in my own way, in line with my understanding and coding style








