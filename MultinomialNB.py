# X: frequency of each words occurs in each document
# Y: spam(1) / notspam(0)
# pi: dirichlet(alpha) prior
# theta
# preprocessing dataset
import pandas as pd

# reading SMS-spam data set
path = '/Users/davidgao/Downloads/smsspamCollection/SMSSpamCollection'
df = pd.read_table(path, sep='	', header=None)
df.columns = ["label", "feature"]
df['isspam'] = df.label.map({'ham': 0, 'spam': 1})

df['length'] = df['feature'].map(lambda text: len(text))
# Stemming & Lemmatization using Python
import nltk

nltk.download('wordnet')
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer

lemmatizer = WordNetLemmatizer()
# 'moved' & 'moving' -> 'move'
df['text_lemmatized'] = df['feature'].map(
    lambda text: ' '.join(lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text.lower())))
import numpy as np

df = df.sample(frac=1)
X = df['text_lemmatized']
Y = df['isspam']  # target 1d
import numpy as np
# remove stop words from text and convert the tedxt content into numrical feature vectors
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(lowercase=True, stop_words='english')

documents = np.array(X)
# learn the vocabulary dictionary and return document-term matrix
documents_counts = count_vect.fit_transform(documents)
X_feature_names = count_vect.get_feature_names()
X = documents_counts.toarray()
Y = Y.to_numpy()


class Multinomial_NB:  # Bayesian approach:dirichlet distribution
    def __init__(self, length, alpha=1.0):  # assume doc uniform distribution
        alhp = np.ones((length, 1))
        alhp.fill(alpha)
        self.__alpha = alhp

    def fit(self, X, Y):
        self.__piorTheta = {
            'SPAM_Likelihood': np.array([]),
            'HAM_Likelihood': np.array([]),
            'feature_names': X_feature_names,
            'spam_word': np.zeros((len(X[0]), 1)),  # frequency of each words appears and it's belongs to a spam email
            'ham_word': np.zeros((len(X[0]), 1))  # frequency of each words appears and it's belongs to a ham email
        }
        # 1. get pior pi pior theta(1d array of sum)
        i = 0
        for email_words_vector in X:  # frequency of each words
            if Y[i] == 0:  # update ham words vector
                self.__piorTheta['ham_word'] = np.add(self.__piorTheta['ham_word'], email_words_vector)
            elif Y[i] == 1:  # update spam words vector
                self.__piorTheta['spam_word'] = np.add(self.__piorTheta['spam_word'], email_words_vector)
            i += 1

        # probability of each word occur in spam emails:
        self.__piorPi = np.ones((len(X), 1))
        piorPiVal = np.count_nonzero(Y) / len(Y)  # pior for spam email(just a number)
        self.__piorPi.fill(piorPiVal)  # (1d array with same values of prob of spam for each email)
        total_hamword = np.sum(self.__piorTheta['ham_word'])
        total_spamword = np.sum(self.__piorTheta['spam_word'])
        self.__piorTheta['HAM_Likelihood'] = np.true_divide(self.__piorTheta['ham_word'],
                                                            total_hamword)  # [letter1 / n_words , letter2 / n_words ...]
        self.__piorTheta['SPAM_Likelihood'] = np.true_divide(self.__piorTheta['spam_word'], total_spamword)

    def predict(self, X):

        probability = self.__piorPi * self.__piorTheta['SPAM_Likelihood']

        return self.predict_log_proba(probability)

    def predict_log_proba(self, X):
        return np.log(X)

    def predict_proba(self, X):
        return self.predict(X)


#########################################
import math


def partition(features, target, t):
    trainingSize = math.floor(len(features) * t)
    testSize = len(features) - trainingSize

    trainingFeature = np.array([])
    trainingTarget = np.array([])
    testFeature = np.array([])
    testTarget = np.array([])

    index = 0
    for row in features:
        if index < trainingSize:
            feature = np.array(row)
            TraintargetValue = target[index]
            if trainingFeature.size == 0:
                trainingFeature = np.append(trainingFeature, feature)
            else:
                trainingFeature = np.vstack((trainingFeature, feature))
            trainingTarget = np.append(trainingTarget, TraintargetValue)
            index += 1
        else:
            feature = np.array(row)
            TesttargetValue = target[index]
            if testFeature.size == 0:
                testFeature = np.append(testFeature, feature)
            else:
                testFeature = np.vstack((testFeature, feature))
            testTarget = np.append(testTarget, TesttargetValue)
            index += 1
    return trainingFeature, trainingTarget, testFeature, testTarget


X_train, X_test, y_train, y_test = partition(X, Y, 0.8)


#########################################
def sPartition(k, data, labels):
    remainder = len(data) % k
    if remainder == 0:
        kfoldsData = np.vsplit(data, k)
        kfoldsLabel = np.split(labels, k)
    else:
        dataReadysplit = data[:-remainder, :]  # remove last n rows
        labelReadysplit = labels[:-remainder]
        kfoldsData = np.vsplit(dataReadysplit, k)
        kfoldsLabel = np.split(labelReadysplit, k)
        remainData = data[-remainder:]
        remainLabel = labels[-remainder:]
        kfoldsData[-1] = np.vstack((kfoldsData[-1], remainData))
        kfoldsLabel[-1] = np.append(kfoldsLabel[-1], remainLabel)
    return kfoldsData, kfoldsLabel


#########################################
# X_train X_test y_train y_test X_feature_names
k = 5
kfoldsData, kfoldsLabel = sPartition(k, X_train, X_test)
# print(kfoldsData)  # #of occurance of each vocab
# print("====================")
# print(kfoldsLabel)  # 0 ham 1spam
########################################

# evaluation(k-fold) to get optimal model

alpha = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0]
# kfoldsData: list
# kfoldsLabel: list

for alphaval in alpha:

    for i in range(0, len(kfoldsData) - 1):

        model = Multinomial_NB(length=len(kfoldsData), alpha=alphaval)

        for j in range(0, len(kfoldsData) - 1):
            if j == i:
                model.fit(kfoldsData[j], kfoldsLabel[j])
            else:
                spamProb = model.predict(kfoldsData[j])
                print(spamProb)
                print(kfoldsLabel[j])
