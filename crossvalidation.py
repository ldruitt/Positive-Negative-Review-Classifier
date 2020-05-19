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
    # for each test point
    for i in range(dists-1):
        pointdists = distarr[:, i]
        pointdists = append_fields(pointdists, 'scores', scores, usemask=False)
        #get distances sorted smallest to largest
        sorted = np.sort(pointdists)
        #get k smallest classifiers (+1 or -1) from the sorted list
        nearest.append(sorted[:k]['scores'])
    # list of each test points k nearest neghbors as their classifiers
    return nearest

def main():
    #get all raw data
    trainfile = open('trainhw1.txt ', 'r')
    train = trainfile.readlines()
    trainfile.close()
    #from the data get the score category for each line (+ or - 1)
    scores = []
    for line in train:
        x = line[:2]
        scores.append(x)
    #change to numpy arrays
    train = np.array(train)
    scores = np.array(scores)

    #split data 10 fold
    train1, train2, train3, train4, train5, train6, train7, train8, train9, train10 = np.array_split(train, 10)
    score1, score2, score3, score4, score5, score6, score7, score8, score9, score10 = np.array_split(scores, 10)

    #normalize the text
    normaltext1 = norm_text(train1)
    normaltext2 = norm_text(train2)
    normaltext3 = norm_text(train3)
    normaltext4 = norm_text(train4)
    normaltext5 = norm_text(train5)
    normaltext6 = norm_text(train6)
    normaltext7 = norm_text(train7)
    normaltext8 = norm_text(train8)
    normaltext9 = norm_text(train9)
    normaltext10 = norm_text(train10)

    #array of arrays
    normals = [normaltext1,normaltext2,normaltext3,normaltext4,normaltext5,normaltext6,normaltext7,normaltext8,normaltext9,normaltext10]
    scorearr = [score1, score2, score3, score4, score5, score6, score7, score8, score9, score10]
    classifile = open('classifications.txt', 'w')
    totalacc = 0
    #crossvalidation, use each of 10 arrays as test value and get accuracy
    for i, normalarr in enumerate(normals):
        traindata = []
        vectordata = []
        for j, normarr in enumerate(normals):
            #get current piece of train data to be used as the test
            if np.array_equal(normalarr,normarr):
                for elem in normalarr:
                    ss = sid.polarity_scores(elem)
                    format = []
                    for k in sorted(ss):
                        format.append(ss[k])
                    vectordata.append(format)
                continue
            #get the rest of the training data to be used as training for the test
            else:
                for elem in normarr:
                    ss = sid.polarity_scores(elem)
                    format = []
                    for k in sorted(ss):
                        format.append(ss[k])
                    traindata.append(format)
                if j < 9:
                    continue
            traindata = np.array(traindata)
            vectordata = np.array(vectordata)

            #calculate the distances for each test point to each train point
            dists = getdistances(vectordata, traindata)

            #each train point is rejoined with its scores
            scorekey = scorearr
            #the scores being removed are the current test scores to be used later to get accuracy
            crossval = np.array(scorearr[i])
            scorekey = np.delete(scorekey, i, 0)
            scorekey = np.concatenate(scorekey, axis=None)
            dists = np.column_stack((dists, scorekey))

            accurate = []
            #get k nearest neighbors
            neighbors = knearest(dists, 15)

            for elem in neighbors:
                #get number of +1s in the list
                numpos = np.sum(elem == '+1')
                #if the majority of the list is positive then classify this point as positive
                if numpos > len(elem)//2:
                    classifile.write('+1\n')
                    accurate.append('+1')
                #otherwise classify as negative
                else:
                    classifile.write('-1\n')
                    accurate.append('-1')
        #compare the predicted to the actual and add the percentage to the sum
        accurate = np.array(accurate)
        correct = (np.sum(accurate == crossval))
        totalacc += correct/len(accurate)
    #get final accuracy for all 10 folds
    print("Accuracy is: " + str(totalacc/10))
    classifile.close()
#    exit(0)


if __name__ == '__main__':
    main()