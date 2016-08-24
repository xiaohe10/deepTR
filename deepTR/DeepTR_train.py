'''
# Implementation of DeepTR which is a very efficient method of predicting term weights, which enables it to be used in online sevices wher lantency must be kept low.
# author : t-hexiao@microsoft.com
# paper(in /doc/): Learning to reweight terms with distributed representations
'''
print (__doc__)

#import basic modules
import os,logging
import re
import warnings
import math
# import modules for word2vector
import gensim
# import modules for lasso regression and matplot
import time

import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
#
# from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
# from sklearn import datasets

from sklearn import linear_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def loadTrainData(file,datacount):
    f = open(file)
    queries = []
    stopwords = loadStopWords('input/stopword.in')
    # delete non-English words
    # delete stopwords
    # todo: use textrank to summarize answers
    pattern = re.compile(r'^[A-Za-z]+$')
    count = 0
    print "loading train data"
    for line in f.readlines():
        print "load raw line:", count
        count += 1
        if (count == datacount): break
        items = line.split("\t")
        query = items[0].split(" ")
        answer = items[1].split(" ")
        tag  = items[2].replace("\n","")
        if(tag == "0"): continue;
        newquery = []
        for que in query:
            if pattern.match(que):
                if (que not in stopwords):
                    newquery.append(que)
        newanswer = []
        for ans in answer:
            if pattern.match(ans):
                if (ans not in stopwords):
                    newanswer.append(ans)
        queries.append({'query':newquery,'answer':newanswer,'origin':items[0],'doc':items[1]})
    f.close()
    print 'all data loaded'
    return queries
def loadStopWords(file):
    f = open(file)
    stpw = []
    for w in f.readlines():
        stpw.append(w.replace('\r\n',''))
    f.close()
    return stpw

# input: query and answers
# use word2vec and lasso regression
# output: query terms weight
def train(datacount,model):
    queries = loadTrainData('input/eqna.tsv',datacount)

    X = []
    Y = []
    print 'start train data'
    count = 0
    all = len(queries)
    for q in queries:
        count += 1

        print count,'/',all
        z_i = []
        all_z = 0
        y_i = []
        x_i = []
        # print q['origin']
        # print q['doc']
        # print len(q['query'])
        normal = 0
        for term in q['query']:
            termweight = 0
            termisvalid = False
            for wordInAnswer in q['answer']:
                try:
                    w = model.similarity(term,wordInAnswer)
                    termweight += w
                    termisvalid = True
                except KeyError:
                    #print "key Error", term, wordInAnswer
                    pass
                #print " %s & %s: %f" % (term, wordInAnswer,w)
            if(termisvalid):
                # if(termweight<=0):
                #     termweight = 0.000001
                # normal += termweight
                termweight /= len(term)
                z_i.append(termweight)
                all_z += termweight
                x_i.append(model[term])
            else:
                pass
        if(all_z == 0):all_z = 1
        for z_i_j in z_i:
            y_i.append(z_i_j/all_z)
                #print "term  invalid:",term
        # get x_i by average and sub
        if(len(x_i) > 0):
            num = len(x_i)
            p = len(x_i[0])
            x_average = [0 for n in range(p)]
            for x_i_j in x_i:
                for j in range(p):
                    x_average[j] += x_i_j[j]/num

            for j in range(len(x_i)):
                for k in range(p):
                    x_i[j][k] -= x_average[k]
        Y.extend(y_i)
        X.extend(x_i)
        #print 'train', q['origin']
    # print X
    # print Y


    # generate true term weight r_ij by word vector
    alphas = [0, 0.000001,0.00001,0.0001, 0.001, 0.01, 0.1,1]
    for alpha in alphas:
        clf = linear_model.LassoLars(alpha  = alpha)
        clf.fit(X, Y)
        print "alpha is ",alpha,":"
        print(clf.coef_)
        write(clf.coef_.tolist(),"output/%d_alphais_%s.in" % (datacount,str(alpha)))
    return clf.coef_
import json

def write(arr,file):
    with open(file, 'w') as outfile:
        json.dump(arr, outfile)
        return True
    return False
def load(file):
    with open(file) as data_file:
        data = json.load(data_file)
        return data
    return None
if __name__ == "__main__":
    #load all queries and answers to train
    print "loading word2 vector model..."
    model = gensim.models.Word2Vec.load_word2vec_format('input/GoogleNews-vectors-negative300.bin',binary=True)
    print "loading over"
    # model = gensim.models.Word2Vec.load('lib/word2vec/models/model.bin')
    B = train(800000,model)
    pass






