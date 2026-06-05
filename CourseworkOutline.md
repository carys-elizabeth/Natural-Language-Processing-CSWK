# Natural-Language-Processing-CSWK
Repository for coursework completed for a Natural Language Processing module 


1. Part One — Syntax and Style
In the first part of your coursework, your task is to explore the syntax and style of a set of
19th Century novels using the methods and tools that you learned in class.
The texts you need for this part are in the texts directory in the coursework Moodle tem-
plate. The texts are in plain text files, and the filenames include the title, author, and year
of publication, separated by hyphens. The template code provided in PartOne.py includes
function headers for some sub-parts of this question. Your finished script should call each
of these functions in order. To complete your coursework, complete these functions so that
they perform the tasks specified in the questions below. You may (and in some cases should)
define additional functions.
(a) 3
read novels: Each file in the texts directory contains the text of a novel, and
the name of the file is the title, author, and year of publication of the novel, separated
by hyphens. Write apython function read novels to do the following:
i. ii. create a pandas dataframe with the following columns: text, title, author, year
sort the dataframe by the year column before returning it, resetting or ignoring
the dataframe index.
(b) 3
nltk ttr: This function should return a dictionary mapping the title of each novel
to its type-token ratio. Tokenize the text using the NLTK library only. Do not include
punctuation as tokens, and ignore case when counting types.
(c) 4
flesch kincaid: This function should return a dictionary mapping the title of
each novel to the Flesch-Kincaid reading ease score of the text. Use the NLTK library
for tokenization and the CMU pronouncing dictionary for estimating syllable counts.
(d) 6
parse: The goal of this function is to process the texts with spaCy’s tokenizer and
parser, and store the processed texts. Your completed function should:
i. Use the spaCy nlp method to add a new column to the dataframe that contains
parsed and tokenized Doc objects for each text.
ii. Serialise the resulting dataframe (i.e., write it out to disk) using the pickle for-
mat.
iii. Return the dataframe.
iv. Load the dataframe from the pickle file and use it for the remainder of this
coursework part. Note: one or more of the texts may exceed the default maxi-
mum length for spaCy’s model. You will need to either increase this length or
parse the text in sections.
(e) Working with parses: the final lines of the code template contain three for loops.
Write the functions needed to complete these loops so that they print:
i. 3
The title of each novel and a list of the ten most common syntactic subjects
overall in the text.
ii. 3
The title of each novel and a list of the verbs most likely to occur with the
subject ‘he’, ordered by their Pointwise Mutual Information.
COIY064H7 Page 2 of 5 ©Birkbeck College 2026
iii. 3
The title of each novel and a list of the verbs most likely to occur with the
subject ‘she’, ordered by their Pointwise Mutual Information.
(f) 5
Five marks are allocated for your git commit history. You should make regular, atomic
commits with concise but informative commit messages. See the section titled Sub-
mission for more details.
Part One total marks: 30
2. 10
Part Two — Feature Extraction and Classification
In the second part of the coursework, your task is to train and test machine learning classifiers
on a dataset of political speeches. The objective is to learn to predict the political party from
the text of the speech.
Your final script should print out the answers to each part where required.
(a) 8
Read the hansard10000.csv dataset in the texts directory into a dataframe. Sub-
set and rename the dataframe as follows:
i. rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’, and then:
ii. remove any rows where the value of the ‘party’ column is not one of the four
most common party names, and remove the ‘Speaker’ value.
iii. iv. remove any rows where the value in the ‘speech class’ column is not ‘Speech’.
remove any rows where the text in the ‘speech’ column is less than 1000 char-
acters long.
Print the dimensions of the resulting dataframe using the shape method.
(b) Vectorise the speeches using TfidfVectorizer from scikit-learn. Use the default pa-
rameters, except for omitting English stopwords and setting max features to 3000.
Split the data into a train and test set, using stratified sampling, with a random seed of
26. Then, train RandomForest (with n estimators=300) and SVM with linear
kernel classifiers on the training set, and print the scikit-learn macro-average f1 score
and classification report for each classifier on the test set. The label that you are trying
to predict is the ‘party’ value.
(c) 5
Adjust the parameters of the Tfidfvectorizer so that unigrams, bi-grams and
tri-grams will be considered as features, limiting the total number of features to 3000.
Print the classification report as in 2(c) again using these parameters.
(d) Implement a new custom tokenizer and pass it to the tokenizer argument of
Tfidfvectorizer. You can use this function in any way you like to try to achieve
the best classification performance while keeping the number of features to no more
than 3000, and using the same three classifiers as above. Print the classification re-
port for the best performing classifier using your tokenizer. Marks will be awarded
both for a reasonable overall classification performance, and a good trade-off between
classification performance and efficiency (i.e., using fewer parameters).
10
COIY064H7 Page 3 of 5 ©Birkbeck College 2026
(e) 7
Explain your tokenizer function and discuss its performance (text answer, 500 words
maximum).
(f) 5
Five marks are allocated for your git commit history. You should make regular, atomic
commits with concise but informative commit messages. See the section below titled
Submission for more details.
Part Two total marks: 45
3. Part Three — Zero-shot and Few-shot Classification with LLMs
In this part of the coursework, you should perform the same political party classification task
as in Part Two, but using prompting with a large language model instead of the scikit-learn
classifiers. For this part use the smaller sample of the speech data in hansard500.csv
You may choose any open-weight model that is available to you through Openrouter, Ollama,
or Huggingface. We will explore these models in class in Week 7. Use the same label set as
in Part Two. Unless a question says otherwise, you should use the same train/test split as in
Part Two.
(a) 4
State which model you used, whether you accessed it through Ollama or OpenRouter,
and any generation parameters that you set. Briefly explain why you chose these
settings.
(b) 8
Implement a zero-shot classification script for the party prediction task. Your prompt
should instruct the model to output one label only. Report the exact zero-shot prompt
that you used, and print the macro-average f1 score and classification report on the
test set.
(c) 8
Implement few-shot classification for the same task. Construct a prompt that includes
a small number of labelled examples from the training data. Report the exact few-shot
prompt that you used, explain how you selected the examples, and print the macro-
average f1 score and classification report on the test set.
(d) 5
Compare the zero-shot and few-shot results. Which performed better? How did you
optimize your prompts? (Text answer, 500 words maximum).
Part Three total marks: 25
