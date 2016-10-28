import gensim
import json
import math

def loadDeepTRModel(file):
    with open(file) as data_file:
        data = json.load(data_file)
        return data
    return None
def sigmoid(x):
  return math.exp(x) / (1 + math.exp(x))


if __name__ == "__main__":
    print "loading word2vec model..."
    model =  gensim.models.Word2Vec.load_word2vec_format('input/GoogleNews-vectors-negative300.bin',binary=True)
    print "model loaded"
    feature_weights = loadDeepTRModel("output/threshold_visualize/80000_alphais_0.in")
    print feature_weights
    while(True):
        query = raw_input("query:")
        if(query == "exit"): break
        items = query.split(" ")
        terms = []
        term_vectors = []
        for item in items:
            try:
                term_vectors.append(model[item])
                terms.append(item)
            except KeyError:
                print "key Error", item
                pass
        if (len(terms) > 0):
            num = len(terms)
            word2vec_dimensions = len(terms[0])
            query_vector_avarage = [0 for n in range(word2vec_dimensions)]
            for termvector in term_vectors:
                for j in range(word2vec_dimensions):
                    query_vector_avarage[j] += termvector[j] / num

            for j in range(len(term_vectors)):
                for k in range(word2vec_dimensions):
                    term_vectors[j][k] -= query_vector_avarage[k]
            # get term weighthow many cigarettes per pack
            term_features = term_vectors
            for i in range(len(term_features)):
                p = len(term_features[0])
                weight = 0
                for j in range(p):
                    weight += term_vectors[i][j] * feature_weights[j]
                weight *= 10
                print terms[i], ":", sigmoid(weight),":",weight
        else:
            print "no valid term"