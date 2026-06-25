import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import pipeline
import torch


#pre-model cleaning of data, not associated with specific question.
directory_path = "/Users/caryswilliams/Documents/Masters Degree Folder/Coursework Pack NLP/texts/hansard500.csv" #amend as required 

def read_and_amend_csv(directory_path): #carried out for cleaning purposes to make train_test_split more consistent for provision of few shot example - almost the same as in Q2, however modified due to length difference.
    df = pd.read_csv(directory_path)
    listed_parties = []
    parties = {}
    listed_parties_and_count = []
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
    listed_parties_and_count.sort(key=lambda x: x[1], reverse = True)#sorting said list of tuples  
    listed_parties_and_count = listed_parties_and_count[4:] #only including those with less mentions than the top 4 
    for item in listed_parties_and_count:
        listed_parties.append(item[0])
    df = df[df.party.isin(listed_parties) == False] #dataframe re-written, only including the top 4 parties
    return df

speeches_dataframe = read_and_amend_csv(directory_path)
speech = speeches_dataframe['speech']
party = speeches_dataframe['party']

speech_train, speech_test, party_train, party_test = train_test_split(speech, party, random_state= 42, train_size=0.7, stratify = party, shuffle=True) #same T/T split as in Q2

#3.a:
#I have chosen to use the Huggingface, "MoritzLaurer/deberta-v3-large-zeroshot-v2.0" as the model for the zeroshot task. There are very few generation settings to chose from, however I have selected multi_label as False, to get the model to 
#respond with one class instead of multiple, in addition to being required in the question, this adds consistency as I understand each 'tree' in the RF classifier in Question two would have also been selecting 1 tree. I used this model as 
#it was easy to understand and interpret the results for, as well as being low demand on my personal computer


#I have selected 4 different speeches from the testing dataset, ensuring that there was one from each party to help "train" the model to see the different examples, whilst aware of the fact that this is not changing any weight within the model's
#overall training process. I selected the last one out of the dataframe, as the first one for one of the parties had a speech that was very long relative to the other speeches

#3.b.
def zero_shot_prompt(text):
    hypothesis_template = "This speech was given by {}"
    classes_verbalized = ['Conservative', 'Labour', 'Democratic Unionist Party', 'Scottish National Party']
    zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")  
    output = zeroshot_classifier(text, classes_verbalized, hypothesis_template=hypothesis_template, multi_label=False)
    return output

#3.c.
def few_shot_prompt(text):
    train_dataframe = pd.DataFrame({"speech": speech_train , "party" : party_train}) 
    prompt = "Here are some speeches with the respective party that made them:\n"
    examples = train_dataframe.groupby('party').last().reset_index(0)
    for _,row in examples.iterrows():
        prompt += f"Question: Who made this Speech: {row['speech']} \n Answer: {row['party']}\n"

    few_shot_template = f"{prompt}. Question: Who made this Speech: {text }""Answer: "
    return few_shot_template 

def few_shot_pipeline(speech_test):
    generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")

    prompt = few_shot_prompt(speech_test)

    output = generator(prompt, max_new_tokens = 5, do_sample=False)

    return output[0]["generated_text"]
    
#3.d.



if __name__ == "__main__":
    party_predict_zero = []
    #for speech in speech_test:
        #output_zero = zero_shot_prompt(speech)
        #party_prediction = output_zero['labels']
        #party_prediction = party_prediction[0]
        #party_predict_zero.append(party_prediction)
    
    party_predict_few = []
    for speech in speech_test:
        output_few = few_shot_pipeline(speech)
        print(output_few)
        break
        
    #consider matplotlib for conf matrx
    #print("Classification Report for Zero Shot")
    #print(classification_report(party_test, party_predict_zero, zero_division=0))
    #print("Macro F1 Score for Zero Shot")
    #print(f1_score(party_test, party_predict_zero, average='macro'))


#AI Declaration
#Built in CoPilot called upon for:
#Install errors for HuggingFace models attempted for few-shot classification








