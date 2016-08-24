import gensim
import json
import math
def loadStopWords(file):
    f = open(file)
    stpw = []
    for w in f.readlines():
        stpw.append(w.replace('\r\n',''))
    f.close()
    return stpw
def loadDeepTRModel(file):
    with open(file) as data_file:
        data = json.load(data_file)
        return data
    return None
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


if __name__ == "__main__":
    stopwords = loadStopWords('input/stopword.in')
    model =  gensim.models.Word2Vec.load_word2vec_format('input/GoogleNews-vectors-negative300.bin',binary=True)
    B = loadDeepTRModel("output/1000000_alphais_0.in")
    print B
    while(True):
        query = raw_input("query:")
        if(query == "exit"): break
        items = query.split(" ")
        terms = []
        w = []
        for item in items:
            try:
                if(item in stopwords):
                    continue
                w.append(model[item])
                terms.append(item)
            except KeyError:
                # print "key Error", term, wordInAnswer
                pass
        if (len(w) > 0):
            num = len(w)
            p = len(w[0])
            w_avarage = [0 for n in range(p)]
            for wi in w:
                for j in range(p):
                    w_avarage[j] += wi[j] / num

            for j in range(len(w)):
                for k in range(p):
                    w[j][k] -= w_avarage[k]
            # get term weighthow many cigarettes per pack
            for i in range(len(w)):
                p = len(w[0])
                weight = 0
                for j in range(p):
                    weight += w[i][j] * B[j]
                print terms[i], ":", sigmoid(weight)
        else:
            print "no valid term"