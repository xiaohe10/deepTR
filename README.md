#DeepTR

[DeepTR](http://www.cs.cmu.edu/~callan/Papers/sigir15-gzheng.pdf) reweights query terms based on the Distributed Representations and can be used in online services where lantency must be kept low.

**Enviroments Requirements:**

Linux + python2.7

python libs: [Anaconda 4.1.1](https://www.continuum.io/downloads "Anaconda 4.1.1") |
[gensim](https://radimrehurek.com/gensim/) | [sciki-learn](http://scikit-learn.org/)

**inputs files**

Pre-trained word and phrase vectors: [GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

Labeled true query results(only access in ms internal): [\\minint-ep6ptdv\Shared\Malta_training_data\eqna_raw_judgments_20151222_norm_train.tsv](\\minint-ep6ptdv\Shared\Malta_training_data\eqna_raw_judgments_20151222_norm_train.tsv) 

please save these input files in deepTR/input as 

> deepTR/input/GoogleNews-vectors-negative300.bin
> 
> deepTR/input/eqna.tsv


**output files**

DeepTR generates a feature weights vector B:

B = [b1, b2, b2, ... , bn]

n is the dimensions of query term word2vector features.

> deepTR/output/1000000\_alphasis_1e-07.in

1000000 is the numbers of count data, 1e-07 is the paramater alpha for lasso regulation


**demo on jupeter notebook**

> [https://172.23.148.90:8888/notebooks/DeepTR.ipynb](https://172.23.148.90:8888/notebooks/DeepTR.ipynb)

> password: Yuze8023
