import nltk, re
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
import sys
import os
import time
import getopt
import pandas as pd
import codecs
import numpy
import csv
from traceback import print_exc


chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', ';',
         '?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-',
         '`']
porter = nltk.PorterStemmer() # also lancaster stemmer
wnl = nltk.WordNetLemmatizer()
stopWords = stopwords.words("english")
word_count_threshold = 5
dataset = 'scdb' # scdb or courtlistener
outputf = 'out'
data_dir = ''
vocabf = ''


def stem(word):
   regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
   stem, suffix = re.findall(regexp, word)[0]
   return stem

def tokenize_text(line):
    if type(line) == float:
        raw = ''
    else:
        raw = line.decode('latin1')
    # raw = ' '.join(raw.rsplit()[1:-1])
    # remove noisy characters; tokenize
    # convert some punctuation to periods for marking negation
    raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
    try:
        tokens = word_tokenize(raw)
        tokens = [w for w in tokens if not re.search('\.', w)]
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if w not in stopWords]
        tokens = [w for w in tokens if not re.search('[0-9]+', w)]
        tokens = [wnl.lemmatize(t) for t in tokens]
        tokens = [w for w in tokens if len(w) > 4]
        tokens = [porter.stem(t) for t in tokens]
    except Exception as e:
        print_exc()
        print tokens
    return tokens

def tokenize_corpus(traintxt, train=True, negation=True, n=1):
    # rewrite so that instead of following path it takes dataframe?
    classes = []
    samples = []
    docs = []
    num_rows = len(traintxt)
    increment = max(500, num_rows / 20)
    if train == True:
        words = {}

    for i, row in enumerate(traintxt.values):
        classes.append('%.0f' % row[-1])
        samples.append(row[0])
        tokens = tokenize_text(row[1])

        if n > 1:
            ngram_tokens = ngrams(tokens, n)
            tokens = [' '.join(ngram) for ngram in ngram_tokens]
            #print 'tokens', tokens
        if train == True:
            for t in tokens:
                try:
                    words[t] = words[t]+1
                except:
                    words[t] = 1
        docs.append(tokens)
        if i % increment == 0:
            print i, '/', num_rows
        # if i == num_rows / 2:
        #     return (docs, classes, samples, words)

    if train == True:
        return(docs, classes, samples, words)
    else:
        return(docs, classes, samples)

def wordcount_filter(words, num=5):
    keepset = []
    for k in words.keys():
        if(words[k] > num):
            keepset.append(k)
    print ("Vocab length:", len(keepset))
    return(sorted(set(keepset)))


def find_wordcounts(docs, vocab):
   bagofwords = numpy.zeros(shape=(len(docs),len(vocab)), dtype=numpy.uint8)
   vocabIndex={}
   for i in range(len(vocab)):
      vocabIndex[vocab[i]]=i

   for i in range(len(docs)):
       doc = docs[i]

       for t in doc:
          index_t=vocabIndex.get(t)
          if index_t>=0:
             bagofwords[i,index_t]=bagofwords[i,index_t]+1

   print( "Finished find_wordcounts for:", len(docs), "docs")
   return(bagofwords)

def construct_docs(X_name, Y_name):
    # construct appropriately formatted docs from X & Y
    # proper format is sample text-string class
    X = pd.read_csv(X_name)
    Y = pd.read_csv(Y_name)
    train = pd.merge(X, Y, left_on='voteId', right_on='voteId')
    text = train.lawMinor.str.cat(' ' + train.caseName + ' ' + train.chief)
    return pd.concat([train.voteId, text, train.partyWinning], axis=1)

def get_vocab(vocabf, traintxt, path=None):
    path = path or data_dir
    global word_count_threshold

    # Tokenize training data (if training vocab doesn't already exist):
    if vocabf == '':
        (docs, classes, samples, words) = tokenize_corpus(traintxt, train=True)
        vocab = wordcount_filter(words, num=word_count_threshold)
        # Write new vocab file
        vocabf = outputf+"_" + dataset + "_vocab_"+str(word_count_threshold)+".txt"
        outfile = codecs.open(path+"/"+vocabf, 'w',"utf-8-sig")
        outfile.write("\n".join(vocab))
        outfile.close()
    else:
        (docs, classes, samples) = tokenize_corpus(traintxt, train=False)
        vocabfile = open(path+"/"+vocabf, 'r')
        vocab = [line.rstrip('\n') for line in vocabfile]
        vocabfile.close()

    print ('Vocabulary file:', path+"/"+vocabf)
    return (docs, classes, samples, vocab)

def get_corpus_scdb(path=None):
    # get corpus for court-centered.
    path = path or data_dir
    trainX = path+"/trainX_text.csv"
    trainY = path+"/trainY.csv"
    testX = path+"/testX_text.csv"
    testY = path+"/testY.csv"
    print ('Path:', path)
    print ('Training data:', trainX, trainY)
    print ('Testing data:', testX, testY)

    train = construct_docs(trainX, trainY)
    test = construct_docs(testX, testY)
    return train, test

def get_corpus_courtlistener(path=None):
    path = path or data_dir
    train = pd.read_csv(path + '/courtlistener.csv')
    return train

def write_csv(bow, name, path=None):
    path = path or data_dir
    with open(path+"/"+outputf+name+str(word_count_threshold)+".csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(bow)

def write_txt(rows, name, path=None):
    path = path or data_dir
    outfile= open(path+"/"+outputf+name+str(word_count_threshold)+".txt", 'w')
    outfile.write("\n".join(rows))
    outfile.close()

def read_args(argv):
    global outputf, data_dir, vocabf, dataset
    try:
        opts, args = getopt.getopt(argv,"d:p:o:v:",
                                   ["dataset=","path=","ofile=","vocabfile="])
    except getopt.GetoptError:
        print ('Usage: \n python preprocess_text.py -d <dataset> '
               '-p <path> -o <outputfile> -v <vocabulary>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('Usage: \n python preprocess_text.py -d <dataset> '
                   '-p <path> -o <outputfile> -v <vocabulary>')
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-p", "--path"):
            data_dir = arg
        elif opt in ("-o", "--ofile"):
            outputf = arg
        elif opt in ("-v", "--vocabfile"):
            vocabf = arg


# fix this.
def main(argv):
    start_time = time.time()
    read_args(argv)

    if dataset == 'scdb':
        (train, test) = get_corpus_scdb()
    else:
        train = get_corpus_courtlistener()
        test = None
    (docs, classes, samples, vocab) = get_vocab(vocabf, train)

    # Get bag of words:
    bow = find_wordcounts(docs, vocab)
    # Check: sum over docs to check if any zero word counts
    print ("Doc with smallest number of words in vocab has:", min(numpy.sum(bow, axis=1)))

    # Write bow file
    prefix = dataset if dataset == 'courtlistener' else 'train'
    write_csv(bow, '_%s_bow' % prefix)

    # Process test.txt / write bow
    if test:
        (docstest, classestest, samplestest) = tokenize_corpus(test, train=False)
        testbow = find_wordcounts(docstest, vocab)
        write_csv(testbow, '_test_bow')
        write_txt(classestest, '_testY')

    # Write classes, samples
    write_txt(classes, '_%s_classes_' % dataset)
    write_txt(samples, '_%s_samples_class_' % dataset)

    print ('Output files:', data_dir+"/"+outputf+"*")

    # Runtime
    print ('Runtime:', str(time.time() - start_time))

if __name__ == "__main__":
    main(sys.argv[1:])
