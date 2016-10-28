# -*- coding: utf-8 -*-
'''
# Implementation of DeepTR which is a very efficient method of predicting term weights, which enables it to be used in online sevices wher lantency must be kept low.
# author : t-hexiao@microsoft.com
# based theory: Learning to reweight terms with distributed representations
'''
print (__doc__)

#import basic modules
import logging
import re
# import modules for word2vector
import gensim
import math
# import modules for lasso regression and matplot
from sklearn import linear_model
import logging
logging.basicConfig(filename='error.log',format='%(message)s',level=logging.ERROR,filemode='w')

threshold = 0.2
# load query and anser terms from eqna data
def loadTrainData(file, datasize):
    f = open(file)
    query_answers = []
    all_term_count = 0
    # delete non-English words
    # delete stopwords
    # todo: use textrank to summarize answers
    pattern = re.compile(r'^[A-Za-z0-9]+$')
    count = 0
    print "loading train data"
    for line in f.readlines():
        count += 1
        print "load raw line:", count
        if (count == datasize): break
        items = line.split("\t")
        query_terms = items[0].split(" ")
        answer_terms = items[1].split(" ")
        flag  = items[2].replace("\n","")
        # flag == 1 means this answer to query is true
        if(flag == "0"): continue;
        else:
            all_term_count += len(query_terms)
            all_term_count += len(answer_terms)
        query_answers.append({'query_terms':query_terms,'answer_terms':answer_terms,'origin_query':items[0]})
    f.close()
    logging.error("all_term_count:"+str(all_term_count))
    print 'all data loaded'
    return query_answers

# input: sample data size, word2vec model loaded globally
# write term weights into files
def train(datasize,model):
    query_answers = loadTrainData('input/eqna.tsv',datasize)

    # all feature vectors of every term in queries, in paper, it is X = (x11, x12, ..., x1n1 , x21, x22, ...x2n2 , ..., xM1, xM2, ..., xMnM)
    all_feature_vectors = []
    # regression labels vector, in paper, it is Y
    all_true_termweight_vectors = []
    print 'start training data'
    count = 0
    alllength = len(query_answers)
    for query_answer in query_answers:
        count += 1
        print count,'/',alllength
        term_vectors = []
        true_term_weights = []
        invalid_terms = []
        for query_term in query_answer['query_terms']:
            termweight = 0
            termisvalid = False
            try:
                model[query_term]
                for answer_term in query_answer['answer_terms']:
                    if(answer_term in invalid_terms): continue
                    try:
                        model[answer_term]
                        similarity = model.similarity(query_term, answer_term)
                        if (similarity > threshold):
                            termweight += similarity
                            termisvalid = True
                    except KeyError:
                        invalid_terms.append(answer_term)
                        logging.error("answer keyerror:" + answer_term)
            except KeyError:
                logging.error("query keyerror:"+query_term)
            # term is valid if we can get this term from word2vec model
            if(termisvalid):
                true_term_weights.append(termweight)
                term_vectors.append(model[query_term])

        # normalize term weight, so their sum will be 1
        sum = 0;
        for termweight in true_term_weights:
            sum += termweight
        if(sum > 0):
            for i in range(0,len(true_term_weights)):
                true_term_weights[i] /= sum
        # get feature vector of a  term: wij − wqi (wij is term word2vec vector , wqi is average term vector)
        average_vector = []
        term_num = len(term_vectors)


        for termvector in term_vectors:
            if not average_vector:
                average_vector = [0]*len(termvector)
            for j in range(0,len(termvector)):
                average_vector[j] += termvector[j]
        for j in range(0, len(average_vector)):
            average_vector[j] /= term_num

        feature_vectors = []
        for i in range(0,len(term_vectors)):
            feature_vectors.append([0]*len(term_vectors[i]))
            for j in range(0, len(term_vectors[i])):
                feature_vectors[i][j] = term_vectors[i][j] - average_vector[j]
        # use logit to map it from (0,1) to real line
        for i in range(0,len(true_term_weights)):
            temp = true_term_weights[i]
            if(temp<= 1e-6):
                temp = 1e-6
            if((1-temp)<=1e-6):
                temp = 1-1e-6
            true_term_weights[i] = math.log(temp/(1-temp))
        all_true_termweight_vectors.extend(true_term_weights)
        all_feature_vectors.extend(feature_vectors)
    # generate true term weight r_ij by word vector
    alphas = [0, 0.000001,0.00001,0.0001, 0.001, 0.01, 0.1,1]
    for alpha in alphas:
        clf = linear_model.LassoLars(alpha  = alpha)
        clf.fit(all_feature_vectors, all_true_termweight_vectors)
        print "alpha is ",alpha,":"
        print(clf.coef_)
        write(clf.coef_.tolist(),"output/threshold/0weight_%d_alphais_%s.in" % (datasize,str(alpha)))
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
    #model = gensim.models.Word2Vec.load('input/model.bin')
    print "loading over"

    B = train(80000,model)
