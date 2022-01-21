"""Name: Jaffarjeet Singh Brar
   Class:  EECS 4412
   Project: Yelp Review Analysis
   Date: Dec 07, 2021
   ID: 215939614
"""

import re
import nltk
import pandas as pd
from nltk import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from nltk.stem.wordnet import WordNetLemmatizer

# load stop words from stopwords.txt that is provided in the project description
stop_words = [(x.strip()) for x in open('stopwords.txt', 'r').read().split('\n')]
classes = ['positive', 'negative', 'neutral']
lmtzr = WordNetLemmatizer()
ps = PorterStemmer()
snow = SnowballStemmer("english")

# # a special tokenizer that removes stop words, converts string to tokens and employs porter stemmer for stemming
# def stemming_tokenizer(input_string):
#     words = re.sub(r"[^A-Za-z0-9\-]", " ", input_string).lower().split()
#     # remove unnecessary characters, convert the string to lower case
#     tokens = list()
#     for word in words:
#         # remove stop words and anything that is not an alphabet
#
#         if word not in stop_words and word.isalpha():
#             tokens.append(word)
#     ps = PorterStemmer()
#     # stemming
#     tokens_stemmed = [ps.stem(w) for w in tokens]
#     return tokens_stemmed


def evaluate_models(models, X_train, X_test, Y_train, Y_test):
    # fit and evaluate the models
    scores_accuracy = list()
    for name, model in models:
        # fit the model
        model.fit(X_train, Y_train)
        ypred = model.predict(X_test)
        acc = accuracy_score(Y_test, ypred)
        scores_accuracy.append(acc)
    # report model performance
    return scores_accuracy


# a function that removes stop words separately for a cleaner data inspection
def discard_stop_words(input_string):
    entire_text = ' '.join(w.lower() for w in input_string.split() if w not in stop_words)

    token = nltk.word_tokenize(entire_text)
    final_output_string = []
    for t in token:

        if t.isalpha():

            t = lmtzr.lemmatize(t)
            if t not in stop_words:
                t = snow.stem(t)
                if t not in stop_words:
                    final_output_string.append(t)
    final_output_string = " ".join(str(s) for s in final_output_string)
    return final_output_string


# df as in data frame created by pandas
def under_sample_positive(df):
    # under sample the positive class csv file named df, create n_samples without replacement
    # and a fixed random state for classifier inspection
    under_sample = resample(df[df['Class'] == 'positive'], n_samples=26000, replace=False, random_state=19)

    # return a new csv file with newly under-sampled_positive, negative and neutral reviews
    return pd.concat([under_sample, df[df['Class'] == 'neutral'], df[df['Class'] == 'negative']])


#                     positive          neutral                    negative

# a function to over sample the minority neutral class
def over_sample_neutral(df):
    over_sample = resample(resample(df[df['Class'] == 'neutral'], n_samples=12000, replace=True, random_state=19))
    return pd.concat([df[df['Class'] == 'positive'], over_sample, df[df['Class'] == 'negative']])


#                       positive                      neutral                    negative

def over_sample_negative(df):
    over_sample = resample(resample(df[df['Class'] == 'negative'], n_samples=15000, replace=True, random_state=19))
    return pd.concat([df[df['Class'] == 'positive'], df[df['Class'] == 'neutral'], over_sample])


#                       positive                      neutral                    negative

df = pd.read_csv("train3.csv")
test_file = pd.read_csv("test3.csv")
df['Text'] = df['Text'].apply(discard_stop_words)
df = over_sample_neutral(df)
df = under_sample_positive(df)
df = over_sample_negative(df)
df.to_csv('Jaffar_training_data.csv')

# define training and testing data with index
X = df['Text']
y = df['Class']
index = df['ID']
x_train, x_test, y_train, y_test, index_train, index_test = train_test_split(X, y, index, train_size=0.8,
                                                                             random_state=19, stratify=y)
# The overall classifier is an ensemble of naive bayes and logistic regression
# First step: build a multinomial naive bayes classifier with TFIDF model
steps = [('vec',
          TfidfVectorizer(strip_accents='unicode',  min_df=10, stop_words=stop_words,
                          ngram_range=(1, 3))), ('mnb', MultinomialNB(alpha=0.1))]
nb = Pipeline(steps)

# Uncomment the following lines to find best feature for multinomial bayes.
# Best features found were min df =10 alpha = 0.1

# # param = {'vec__min_df':[1, 10], 'mnb__alpha':[0.1, 0.01, 0.001]}
# # nb = GridSearchCV(pipeline, param, cv = 10, scoring="accuracy", verbose=1)
steps = [('tfidf',
          CountVectorizer(strip_accents='unicode',  stop_words=stop_words, min_df=2,
                          ngram_range=(1, 3))),
         ('lr', LogisticRegression(n_jobs=-1))]
LR = Pipeline(steps)
models = [('lr', LR), ('nb', nb)]
scores = evaluate_models(models, x_train, x_test, y_train, y_test)
# create the ensemble
ensemble = VotingClassifier(estimators=models, voting='soft', weights=scores)
ensemble.fit(x_train, y_train)
y_predicted = ensemble.predict(x_test)
print("The weighted scores are", scores)
print('Weighted Average accuracy is:  %.3f' % (accuracy_score(y_test, y_predicted) * 100))
print(classification_report(y_test, y_predicted, target_names=classes))
print(confusion_matrix(y_test, y_predicted))

# test_file
# preprocess test file in the same way as training file
test_file['Text'] = test_file['Text'].apply(discard_stop_words)
test_file = test_file.set_index('ID')
test_file['Class'] = ''
y_final_predicted = ensemble.predict(test_file['Text'])
predictions = test_file
predictions['Class'] = y_final_predicted
# This is just temporary file to analyze predictions with results by manual viewing
predictions.to_csv('Analyze_predictions.csv')
# final prediction file in correct format
predictions.index.names = ['REVIEW ID']
predictions = predictions.drop(columns=['Text'])
predictions.to_csv("prediction.csv")
