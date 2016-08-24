import os.path
dir_path = os.path.dirname(os.path.realpath(__file__))
def isFileExist():
    fname = os.path.join(dir_path, "input/GoogleNews-vectors-negative300.bin")
    word2vec =  os.path.isfile(fname)
    fname = os.path.join(dir_path, "input/eqna.tsv")
    eqna =  os.path.isfile(fname)
    if(word2vec and eqna):
        print "input file is ok"
        return True
    else:
        print "please download word2vector model and eqna data according to README.me"
        return False

if __name__ == "__main__":
    isFileExist()