#Question 2 - Feature Extraction and Classification

import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn import svm



#2.a:

#function to read csv file into a dataframe, with certain elements removed (Labour(Co-op) renamed to Labour, Speaker rows removed, parties outside of the most common 4 removed, non speech rows removed and any speech shorter than 1000 characters removed.)

directory_path = "/Users/caryswilliams/Documents/Masters Degree Folder/Coursework Pack NLP/texts/hansard10000.csv" #input own directory path here
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
    return df

#2.b. 
def tfidfVectorizer_RF_SVM(speeches_dataframe): #function to carry out Random Forest and SVM on the speeches dataframe (with vectorized words)
    speech = speeches_dataframe['speech']
    party = speeches_dataframe['party']

    speech_train_pre_vec, speech_test_pre_vec, party_train, party_test = train_test_split(speech, party, stratify = party, train_size=0.7, random_state=26, shuffle = True)
    
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=3000
    )
    training_speech = vectorizer.fit_transform(speech_train_pre_vec)

    testing_speech = vectorizer.fit_transform(speech_test_pre_vec)

    #training the RF classifier 
    RFclass = RandomForestClassifier(n_estimators=300)
    RFclass.fit(training_speech, party_train)
    predictRF = RFclass.predict(testing_speech)

    #macro average F1 score for RF 
    accuracyRF = f1_score(party_test, predictRF, average='macro')
    print(f"Macro Average F1 Score for Random Forest:\n {accuracyRF}")

    #classification report for RF 
    print(f"Classification Report for Random Forest:\n {classification_report(party_test, predictRF)}")

    #training the SVM classifier 
    SVMclass = svm.LinearSVC()
    SVMclass.fit(training_speech, party_train)
    predictSVM = SVMclass.predict(testing_speech)

    #macro average F1 score for SVM
    accuracySVM = f1_score(party_test, predictSVM, average='macro')
    print(f"Macro Average F1 Score for SVM:\n {accuracySVM}")

    #classification report for SVM
    print(f"Classification Report for SVM:\n {classification_report(party_test, predictSVM)}")

#2.c. 
def tfidfVectorizer_bi_tri(speeches_dataframe): #repeat of tfidfVectorizer_RF_SVM as above, but with 2c conditions applied, with uni/bi/tri grams considered
    speech = speeches_dataframe['speech']
    party = speeches_dataframe['party']

    speech_train_pre_vec, speech_test_pre_vec, party_train, party_test = train_test_split(speech, party, stratify = party, train_size=0.7, random_state=26, shuffle = True)
    
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1,3), max_features=3000, 
    )
    training_speech = vectorizer.fit_transform(speech_train_pre_vec)

    testing_speech = vectorizer.fit_transform(speech_test_pre_vec)

    #training the RF classifier 
    RFclass = RandomForestClassifier(n_estimators=300)
    RFclass.fit(training_speech, party_train)
    predictRF = RFclass.predict(testing_speech)

    #macro average F1 score for RF 
    accuracyRF = f1_score(party_test, predictRF, average='macro')
    print(f"Macro Average F1 Score for Random Forest:\n{accuracyRF}")

    #classification report for RF 
    print(f"Classification Report for Random Forest:\n{classification_report(party_test, predictRF)}")

    #training the SVM classifier 
    SVMclass = svm.LinearSVC()
    SVMclass.fit(training_speech, party_train)
    predictSVM = SVMclass.predict(testing_speech)

    #macro average F1 score for SVM
    accuracySVM = f1_score(party_test, predictSVM, average='macro')
    print(f"Macro Average F1 Score for SVM:\n{accuracySVM}")

    #classification report for SVM
    print(f"Classification Report for SVM:\n{classification_report(party_test, predictSVM)}")


#2.d. new custom tokenizer passed to the arg of Tfidfvectorizer
   
def custom_lemm(text): #new custom tokenizer - new inferred to mean, alternate to one used within TfidfVectorizer as opposed to completely new and unique 
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    

def vectorizing_with_custom(speeches_dataframe): 
    speech = speeches_dataframe['speech']
    party = speeches_dataframe['party']
    vectorizer = TfidfVectorizer(ngram_range = (1,4), stop_words= 'english', 
                                 tokenizer=custom_lemm, 
                                 max_features= 3000, token_pattern = None)

    speech_train, speech_test, party_train, party_test = train_test_split(speech, party, random_state= 42, train_size=0.7, stratify = party, shuffle=True)

    training_speech = vectorizer.fit_transform(speech_train)
    testing_speech = vectorizer.fit_transform(speech_test)

    RFclass = RandomForestClassifier(n_estimators=300, min_samples_split=10, max_depth= 30, max_features=3000) #amended to attempt to avoid overfitting
    RFclass.fit(training_speech, party_train)
    predictRF = RFclass.predict(testing_speech)

    #macro average F1 score for RF 
    accuracyRF = f1_score(party_test, predictRF, average='macro')
    print(f"Macro Average F1 Score for Random Forest:\n{accuracyRF}")

    #classification report for RF
    print(f"Classification Report for Random Forest:\n{classification_report(party_test, predictRF, zero_division = 0)}") #zero division set to 0 to avoid undefined prevision when true + false positive == 0

    print("Comparison of Training/Testing Performance:\n")
    print(f"\nFinal Training Accuracy: {RFclass.score(training_speech, party_train)*100}%")
    print(f"Model Accuracy: {RFclass.score(testing_speech, party_test)*100}%")


#2.e: 
#The tokenising model that I have used for the final task was a lemmatising model through NLTK. 
#This involves identifying the “inflected form” of a word and returning this as opposed to the full word. 
#I used this with the intention that it would allow the model to identify themes within party speeches through having a greater picture of the word forms being used, 
#I found that using a model that performed stemming instead had poor performance and often defaulted to assuming all speeches were Conservative 
#(the higher represented party among the sample), likely because lemmatising can be too ‘aggressive’ and leave parts of words that do not make sense. 
#I also added an element to this tokeniser to remove capitalisation of the text in order to allow the model to not be impacted by capitalised words.

#The F1 score is quite poor, at 0.22, this demonstrates poor performance of the precision and recall overall, likely impacted by the poor recall for the other parties and lack of any classification for Liberal Democrats. 
#This is demonstrated throughout the classification report. I note that the recall is high for the Conservative Party, and found from production of a truth table that the model was classifying most speeches as Conservative, 
#which is why the recall score for all other parties is very low. The Scottish national party has high precision, this went from 0.5 to 1 when the text was made lower case, 
#and the model appears to be good at avoiding mistakes when classifying their speeches, however due to the low recall, it appears that the model did not call that party often. 
#The accuracy is okay, it is similar to what the model would score if it called all speeches as Conservative, as they make up approximately 60% of the speeches.

#The Random Forest parameters were modulated to attempt to avoid overfitting, however from the Training and Model Accuracy that were created, it appears that the accuracy in 
#training was nearly 100% and this did not carry through to unseen data, likely indicating further adaptations could have been made to further prevent overfitting and improve accuracy overall.
#The max_depth was especially hard to modulate, I found that if this was any lower than 30 then almost all the model would predict was Conservative.
 

if __name__ == "__main__":
    print("Shape of Adapted Hansard 10,000 Dataframe:\n")
    speeches_dataframe = read_and_amend_csv(directory_path)
    shape = speeches_dataframe.shape
    print(f"{shape}")
    print("\nRandom Forest and SVM on Vectorized Dataframe:\n")
    classif_f1_vec_1 = tfidfVectorizer_RF_SVM(speeches_dataframe)
    print(classif_f1_vec_1)
    print("\nRandom Forest and SVM on Vectorized Dataframe with NGram Range 1-3:\n")
    classif_f1_vec_2 = tfidfVectorizer_bi_tri(speeches_dataframe)
    print(classif_f1_vec_2)
    print("\nRandom Forest for Vectorized Dataframe with Custom Tokenizer:")
    testing = vectorizing_with_custom(speeches_dataframe)
    print(testing)




#AI Declaration:
#Built in Co-pilot tool was used for:
#Install error with "Punkt" - SSL certificate issue on macOS that needed correcting for it to run 
#Error in code where I had placed the wrong variable inside my confusion matrix code.