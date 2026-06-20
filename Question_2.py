#Question 2 - Feature Extraction and Classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.metrics import classification_report


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

def tfidfVectorizer_RF_SVM(speeches_dataframe):
    speech = speeches_dataframe['speech']
    party = speeches_dataframe['party']

    X_train, X_test, y_train, y_test = train_test_split(speech, party, stratify = party, test_size=0.2, random_state=26)
    
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



testing = tfidfVectorizer_RF_SVM(speeches_dataframe)

print(testing)





    


