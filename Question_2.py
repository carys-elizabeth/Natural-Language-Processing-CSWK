#Question 2 - Feature Extraction and Classification

import pandas as pd
import numpy as np
import spacy
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer


#2.a:

#function to read csv file into a dataframe, with certain elements removed (Labour(Co-op) renamed to Labour, Speaker rows removed, parties outside of the most common 4 removed, non speech rows removed and any speech shorter than 1000 characters removed.)

directory_path = "/Users/caryswilliams/Documents/Masters Degree Folder/Coursework Pack NLP/texts/hansard10000.csv"

def read_and_amend_csv(directory_path):
    parties = {}
    listed_parties_and_count = []
    listed_parties = []
    df = pd.read_csv(directory_path)
    notspeech = df[df["speech_class"] != "Speech"].index #removing rows that aren't listed as a speech 
    df.drop(notspeech, inplace = True)
    speaker_row = df[df["party"] == "Speaker"].index #removing rows with the Speaker value as party 
    df.drop(speaker_row, inplace = True)
    df["party"] = np.where(df["party"] == "Labour (Co-op)", "Labour", df["party"])
    for index,row in df.iterrows(): #creating a dictionary counting the number of occurrences of each party 
        party = row["party"]
        parties[party] = parties.get(party, 0) + 1
    for party, count in parties.items(): #creating a list of tuples(sortable) of parties and their count 
        listed_parties_and_count.append((party, count))
    listed_parties_and_count.sort(key=lambda x: x[1], reverse = True) #sorting said list of tuples 
    listed_parties_and_count = listed_parties_and_count[4:] #only including those with less mentions than the top 4 
    for item in listed_parties_and_count:
        listed_parties.append(item[0])
    df = df[df.party.isin(listed_parties) == False] #dataframe re-written, only including the top 4 parties 
    df = df[df.speech.str.len() >= 1000]
    shape = df.shape
    print(f"Shape of the dataframe:{shape}")
    return df

speeches_dataframe = read_and_amend_csv(directory_path)

#2.b. function to carry out Random Forest and SVM on the speeches dataframe (with vectorized words)

def tfidfVectorizer_RF_SVM(speeches_dataframe):
    speech = speeches_dataframe['speech']
    party = speeches_dataframe['party']

    X_train, X_test, y_train, y_test = train_test_split(speech, party, stratify = party, test_size=0.2, random_state=26, shuffle = True)
    
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=3000
    )
    training_speech = vectorizer.fit_transform(X_train)

    testing_speech = vectorizer.fit_transform(X_test)

    #training the RF classifier 
    RFclass = RandomForestClassifier(n_estimators=300)
    RFclass.fit(training_speech, y_train)
    predictRF = RFclass.predict(testing_speech)

    #macro average F1 score for RF 
    accuracyRF = f1_score(y_test, predictRF, average='macro')
    print(f"Macro Average F1 Score for Random Forest: {accuracyRF}")

    #classification report for RF 
    print(f"Classification Report for Random Forest: {classification_report(y_test, predictRF)}")

    #training the SVM classifier 
    SVMclass = svm.LinearSVC()
    SVMclass.fit(training_speech, y_train)
    predictSVM = SVMclass.predict(testing_speech)

    #macro average F1 score for SVM
    accuracySVM = f1_score(y_test, predictSVM, average='macro')
    print(f"Macro Average F1 Score for SVM: {accuracySVM}")

    #classification report for SVM
    print(f"Classification Report for SVM: {classification_report(y_test, predictSVM)}")


#2.c. repeat of tfidfVectorizer_RF_SVM as above, but with 2c conditions applied, with uni/bi/tri grams considered

def tfidfVectorizer_bi_tri(speeches_dataframe):
    speech = speeches_dataframe['speech']
    party = speeches_dataframe['party']

    X_train, X_test, y_train, y_test = train_test_split(speech, party, stratify = party, test_size=0.3, random_state=26, shuffle = True)
    
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1,3), max_features=3000, 
    )
    training_speech = vectorizer.fit_transform(X_train)

    testing_speech = vectorizer.fit_transform(X_test)

    #training the RF classifier 
    RFclass = RandomForestClassifier(n_estimators=300)
    RFclass.fit(training_speech, y_train)
    predictRF = RFclass.predict(testing_speech)

    #macro average F1 score for RF 
    accuracyRF = f1_score(y_test, predictRF, average='macro')
    print(f"Macro Average F1 Score for Random Forest:\n{accuracyRF}")

    #classification report for RF 
    print(f"Classification Report for Random Forest:\n{classification_report(y_test, predictRF)}")

    #training the SVM classifier 
    SVMclass = svm.LinearSVC()
    SVMclass.fit(training_speech, y_train)
    predictSVM = SVMclass.predict(testing_speech)

    #macro average F1 score for SVM
    accuracySVM = f1_score(y_test, predictSVM, average='macro')
    print(f"Macro Average F1 Score for SVM:\n{accuracySVM}")

    #classification report for SVM
    print(f"Classification Report for SVM:\n{classification_report(y_test, predictSVM)}")

#testing1 = tfidfVectorizer_bi_tri(speeches_dataframe)
#print(testing1)

#2.d. new custom tokenizer passed to the arg of Tfidfvectorizer

def custom_token_nltk_lemm(speeches):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(speeches)
    tokens = [token for token in tokens if token.isalpha()]
    lemma = [lemmatizer.lemmatize(word) for word in tokens]
    if len(lemma) > 2:
        return lemma

def vectorizing_with_custom(speeches_dataframe): #seems to only be good if random seed is 100 - otherwise no selection for lib dems +/- scottish np 
    speech = speeches_dataframe['speech']
    party = speeches_dataframe['party']
    parties = ['Conservative', 'Labour', 'Liberal Democrat', 'Scottish National Party']
    vectorizer = TfidfVectorizer(ngram_range = (1,4), stop_words= 'english', 
                                 tokenizer=custom_token_nltk_lemm, 
                                 max_features= 3000, token_pattern = None, smooth_idf=True, sublinear_tf=True)

    speech_train, speech_test, party_train, party_test = train_test_split(speech, party, random_state= 42, train_size=0.7, stratify = party, shuffle=True)

    training_speech = vectorizer.fit_transform(speech_train)
    testing_speech = vectorizer.fit_transform(speech_test)

    RFclass = RandomForestClassifier(n_estimators=300, min_samples_split=10, max_depth= 30, max_features=3000)
    RFclass.fit(training_speech, party_train)
    predictRF = RFclass.predict(testing_speech)

    #macro average F1 score for RF 
    accuracyRF = f1_score(party_test, predictRF, average='macro')
    print(f"Macro Average F1 Score for Random Forest:\n{accuracyRF}")

    #classification report for RF
    print(f"Classification Report for Random Forest:\n{classification_report(party_test, predictRF, zero_division = 0)}") #zero division set to 0 to avoid undefined prevision when true + false positive == 0
    print(f"Final Training Accuracy: {RFclass.score(training_speech, party_train)*100}%")
    print(f"Model Accuracy: {RFclass.score(testing_speech, party_test)*100}%")

    #training the SVM classifier 
    SVMclass = svm.LinearSVC()
    SVMclass.fit(training_speech, party_train)
    predictSVM = SVMclass.predict(testing_speech)

    #macro average F1 score for SVM
    accuracySVM = f1_score(party_test, predictSVM, average='macro')
    print(f"Macro Average F1 Score for SVM:\n{accuracySVM}")

    #classification report for SVM
    #print(f"Classification Report for SVM:\n{classification_report(party_test, predictSVM, zero_division=0)}")

    import matplotlib.pyplot as plt

    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(party_test, predictRF, display_labels=parties, ax=ax)
    _ = ax.set_title(
        f"Confusion Matrix for {predictRF.__class__.__name__}\non the original documents"
    )
    plt.plot() 
    #plt.show()

testing = vectorizing_with_custom(speeches_dataframe)

print(testing)



