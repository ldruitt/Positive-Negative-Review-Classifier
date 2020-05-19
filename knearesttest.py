import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from numpy.lib.recfunctions import append_fields
import re


#tokenizer
punct = nltk.WordPunctTokenizer()
#stopwords
words = nltk.corpus.stopwords.words('english')
words.remove('not')
words.remove('no')
#lemmatization
stemmer = SnowballStemmer("english", ignore_stopwords=True)
#sentiment analyzer
sid = SentimentIntensityAnalyzer()

#preprocessing the text
def normalizetext(text):
    #lemmatize
    text = stemmer.stem(text)
    #remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    #all text to lowercase
    text = text.lower()
    #remove extra whitespace
    text = text.strip()
    #tokenize the text
    tokens = punct.tokenize(text)
    #remove stopwords
    filtered_tokens = [token for token in tokens if token not in words]
    #join final processed tokens
    finaltext = ' '.join(filtered_tokens)
    return finaltext

#vectorize the normalizetext function
norm_text = np.vectorize(normalizetext)

def getdistances(X, X_train):
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_train, num_test))
    #L2 distance formula done with numpy on entire array
    dists = np.sqrt((X ** 2).sum(axis=1)[:, np.newaxis] + (X_train ** 2).sum(axis=1) - 2 * X.dot(X_train.T))
    return dists.T

def knearest(distarr, k):
    dists = distarr.shape[1]
    pointdists = np.zeros((dists, 2))
    scores = np.array(distarr[:, dists-1])
    nearest = []
    #for each test point
    for i in range(dists-1):
        pointdists = distarr[:, i]
        pointdists = append_fields(pointdists, 'scores', scores, usemask=False)
        # get distances sorted smallest to largest
        sorted = np.sort(pointdists)
        # get k smallest classifiers (+1 or -1) from the sorted list
        nearest.append(sorted[:k]['scores'])
    #list of each test points k nearest neghbors as their classifiers
    return nearest


def main():
    #get all raw data
    trainfile = open('trainhw1.txt', 'r')
    train = trainfile.readlines()
    trainfile.close()

    testfile = open('testdatahw1.txt', 'r')
    test = testfile.readlines()
    testfile.close()
    #from the data get the score category for each line (+ or - 1)
    scores = []
    for line in train:
        x = line[:2]
        scores.append(x)
    #change to numpy arrays
    train = np.array(train)
    scores = np.array(scores)


    #normalize the text
    normaltrain = norm_text(train)
    normaltest  = norm_text(test)

    classifile = open('classifications.txt', 'w')

    vectordata = []
    traindata = []
    #get each lines sentiment scores
    for elem in normaltrain:
        ss = sid.polarity_scores(elem)
        format = []
        for k in sorted(ss):
            format.append(ss[k])
        traindata.append(format)
    for elem in normaltest:
        ss = sid.polarity_scores(elem)
        format = []
        for k in sorted(ss):
            format.append(ss[k])
        vectordata.append(format)

    traindata = np.array(traindata)
    vectordata = np.array(vectordata)

    #for memory management i split my data up to be run in batches
    vectordata = np.array_split(vectordata, 10)

    #every batch in the test data array, calculate the nearest neighbors
    for i,smallvector in enumerate(vectordata):
        #calculate the distances for each test point to each train point
        dists = getdistances(smallvector, traindata)
        dists = np.column_stack((dists, scores))
        # get k nearest neighbors for each test point
        neighbors = knearest(dists, 15)
        for elem in neighbors:
            #get number of +1s in the list
            numpos = np.sum(elem == '+1')
            # if the majority of the list is positive then classify this point as positive
            if numpos > len(elem) // 2:
                classifile.write('+1\n')
            # otherwise classify as negative
            else:
                classifile.write('-1\n')

    classifile.close()
#    exit(0)


if __name__ == '__main__':
    main()