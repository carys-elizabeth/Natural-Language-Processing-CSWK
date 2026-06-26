import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import T5Tokenizer, T5ForConditionalGeneration


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
#it was easy to understand and interpret the results for, as well as being low demand on my personal computer.

#for the Few Shot task I chose to use "google/flan-t5-small", accessed through HuggingFace as the initial set up was fairly simple to implement with few parameters required, although it is mainly generated for language translation tasks, 
#it is able to do text generation tasks so I felt it would be able to support with this task, although unfortunately that did lead to some errors in it's response generation. The only parameters that were set within the code for this model was
#return tensors: pt. This was initiated as outlined in the model page on Hugging Face. And max_new_tokens was set as 3, to avoid the model generating large streams of text as it's answer.


#3.b.
#The exact zero shot prompt was "this speech was given by: " where the model would then input the response after, in line with the 4 classes given (Conservative, Labour, DUP and SNP - which were the top 4 left in this dataframe after the initial cleaning took place)
def zero_shot_prompt(text):
    hypothesis_template = "Who was this speech given by? {}"
    classes_verbalized = ['Conservative', 'Labour', 'Democratic Unionist Party', 'Scottish National Party']
    zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")  
    output = zeroshot_classifier(text, classes_verbalized, hypothesis_template=hypothesis_template, multi_label=False)
    return output

#3.c.
#I have selected 2 different speeches from the testing dataset, these were selected as the last 2 speeches of 2 parties from the training dataset, initially I had aimed for one from each party however this led to the length being too long for the model, this was intended on giving the model an idea of what some speech inputs would look like whilst aware of the fact that this is not changing any weight within the model's
#overall training process. I selected the last one out of the dataframe for the two parties as the first one for one of the parties had a speech that was very long relative to the other speeches
#The full prompt was:
#Here are some speeches with the respective party that made them: Question: Who made this Speech? It seems to me that we need to find the right balance between protecting the rights of individuals to rightfully challenge Executive power and ensuring that government can proceed effectively without vexatious legal claims. How will my right hon. and learned Friend ensure that we get the right balance? Answer: Conservative. Question: Who made this Speech? This House is united in its joint calls for our Government to act and respond robustly. I first raised the treatment of the Uyghurs in this House in 2015, yet here we are five years later and the situation remains every bit as desperate. I know it is not the personal responsibility of the Minister, but I believe we have a moral obligation to use whatever channels are available to ensure that all is done to penalise China. We must apply as much pressure as we can to help those who are being persecuted only because of their religion and their faith. Answer: Democratic Unionist Party. Question: Who made this speech {speech_test}. Answer (Either Conservative, Labour, Democratic Unionist Party or Scottish National Party): 

#def few_shot_prompt(text): #This was the initial code used to generate the prompt, which gave a speech from each party to put in the prompt, however this generated speech was too long for the model. So a section of the output of this was taken and given as a prompt to the model.
    #train_dataframe = pd.DataFrame({"speech": speech_train , "party" : party_train}) 
    #prompt = "Here are some speeches with the respective party that made them:.\n"
    #examples = train_dataframe.groupby('party').last().reset_index(0)
    #for _,row in examples.iterrows():
        #prompt += f"Question: Who made this Speech? {row['speech']} \n Answer: {row['party']}\n"

    #few_shot_template = f"{prompt}. Question: Who made this Speech? {text }. Answer: "
    #return few_shot_template 

def few_shot_pipeline(speech_test): #code inputted as per the format recommended on the HuggingFace page for google/flan-t5-small
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    input_text = f"Here are some speeches with the respective party that made them: Question: Who made this Speech? It seems to me that we need to find the right balance between protecting the rights of individuals to rightfully challenge Executive power and ensuring that government can proceed effectively without vexatious legal claims. How will my right hon. and learned Friend ensure that we get the right balance? Answer: Conservative. Question: Who made this Speech? This House is united in its joint calls for our Government to act and respond robustly. I first raised the treatment of the Uyghurs in this House in 2015, yet here we are five years later and the situation remains every bit as desperate. I know it is not the personal responsibility of the Minister, but I believe we have a moral obligation to use whatever channels are available to ensure that all is done to penalise China. We must apply as much pressure as we can to help those who are being persecuted only because of their religion and their faith. Answer: Democratic Unionist Party. Question: Who made this speech {speech_test}. Answer (Either Conservative, Labour, Democratic Unionist Party or Scottish National Party): "
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids, max_new_tokens = 3)
    return(tokenizer.decode(outputs[0]))
    
#3.d.
#The prompt for few shot went through several iterations, which included multiple positions of a list of the specified parties that it would be expected to respond with. The model would often respond with additional parties, and on a few occasions, continue the speech with different text. This is likely a limitation of the type of model chosen, additionally the model may have been impacted by the length of the prompt going over it’s limit, therefore the speeches it was being given were shortened to 400 characters and the number of speeches given in the prompt was reduced from 4 to 2. 
#The output of this model with the current prompt improved significantly, and the model was now only responding with political parties, however the format of it’s responses was mixed, with it sometimes returning “Conservative Party” as opposed to Conservative, and sometimes adding a “.” on the end so it therefore shows up as a different response within the Confusion Matrix.

#Therefore the overall F1 score and Accuracy of the few shot model was found to be 0 from the built in classifier. From a confusion matrix of the response, out of the 94 Conservative speeches it was given, 88 of those were classified as Conservative (through various formatting), for Labour, out of the 32 speeches it was given only 3 were accurately classified (and one was classified as “Conservative, Labour”), for Scottish National Party and Democratic Unionist Part of the 10 ad 2 it was given 0 were accurately classified. This demonstrates that this model has a somewhat similar output to the Random Forest model in Question 2, where it defaults to classifying as Conservative. 

#The prompt for Zero shot was modified from "This speech was given by : {}" to "Who was this speech given by?" and this improved the Macro F1 score from 0.27 to 0.49.
#Likely due to the LLM having a clearer question to answer. 

#The Zero shot model performed well in comparison to the few shot, the overall accuracy was 65%, which is similar to the models trialled in Question 2, but the precision for Conservative and Scottish National party was 82% and 83% respectively, it demonstrates that this model does not fall victim to the tendency to default to the Conservative Party. With the improvement in the Zero shot prompt, the precision, recall and F1 score for the Labour Party increased from 0 to 0.47, suggesting that with the previous prompt this party was not being suggested for any speeches, and that the models’ ability to recognise their speeches improved with the alternative prompt. 

#It is hard to compare the different models fully as they are trained in different ways but when the same model used for few shot was given a zero shot prompt, the model tended to produce varying responses including politician names or words such as “Welsh” and “Minister”. Indicating that this model is not optimised for tasks without examples given or a chain of thought question may have improved output. 

if __name__ == "__main__":
    party_predict_few = []
    party_predict_amended = []
    for speech in speech_test:
        truncated_speech = speech[:400]
        output_few = few_shot_pipeline(speech)
        party_prediction_few = output_few
        party_predict_few.append(party_prediction_few)


    party_predict_zero = []
    for speech in speech_test:
        output_zero = zero_shot_prompt(speech)
        party_prediction = output_zero['labels']
        party_prediction = party_prediction[0]
        party_predict_zero.append(party_prediction)

    
    print("Classification Report for Zero Shot\n")
    print(classification_report(party_test, party_predict_zero, zero_division=0))
    print("\nMacro F1 Score for Zero Shot\n")
    print(f1_score(party_test, party_predict_zero, average='macro'))

    print("\nClassification Report for Few Shot:\n")
    print(classification_report(party_test, party_predict_few, zero_division=0))
    print("\nMacro F1 Score for Few Shot:\n")
    print(f1_score(party_test, party_predict_few, average='macro'))

    cmzero = ConfusionMatrixDisplay.from_predictions(party_test, party_predict_zero)
    plt.show()

    cmfew = ConfusionMatrixDisplay.from_predictions(party_test, party_predict_few)
    plt.show()




#AI Declaration
#Built in CoPilot called upon for:
#Install errors for HuggingFace models attempted for few-shot classification. 
#Support with challenges faced running some few shot models due to limitations of computer
#explanation of some parameters required by the models, due to lack of clarity in their respective HuggingFace pages.

#No code was directly generated and used in Question 3, all suggestions by CoPilot were reviewed in the CoPilot window but adapted and implemented in my own way, in line with my understanding and coding style






