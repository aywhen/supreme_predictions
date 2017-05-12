import nltk, re
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']
porter = nltk.PorterStemmer() # also lancaster stemmer
wnl = nltk.WordNetLemmatizer()
stopWords = stopwords.words("english")

def stem(word):
   regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
   stem, suffix = re.findall(regexp, word)[0]
   return stem

def tokenize_text(line):
    raw = line.decode('latin1')
    # raw = ' '.join(raw.rsplit()[1:-1])
    # remove noisy characters; tokenize
    # convert some punctuation to periods for marking negation
    raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
    tokens = word_tokenize(raw)
    tokens = [w for w in tokens if not re.search('\.', w)]
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in stopWords]
    tokens = [wnl.lemmatize(t) for t in tokens]
    tokens = [porter.stem(t) for t in tokens]
    return tokens

def tokenize_corpus(traintxt, train=True, negation=True, n=1):
    # rewrite so that instead of following path it takes dataframe?
    classes = []
    samples = []
    docs = []
    if train == True:
        words = {}

    for row in traintxt:
        classes.append(row[-1])
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

    if train == True:
        return(docs, classes, samples, words)
    else:
        return(docs, classes, samples)

def wordcount_filter(words, num=5):
   keepset = []
   for k in words.keys():
       if(words[k] > num):
           keepset.append(k)
   print "Vocab length:", len(keepset)
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

   print "Finished find_wordcounts for:", len(docs), "docs"
   return(bagofwords)

def construct_docs(X, Y):
    # construct appropriately formatted docs from X & Y
    # proper format is sample text-string class
    train = pd.merge(p.trainX_text,
                       p.trainY,
                       left_on='voteId',
                       right_on='voteId')
    text = train.lawMinor.str.cat(' ' + train.caseName + ' ' + train.chief)
    return pd.concat([train.voteId, text, train.partyWinning], axis=1)

def get_vocab(vocabf, traintxt):
    # Tokenize training data (if training vocab doesn't already exist):
    if (not vocabf):
        word_count_threshold = 5
        (docs, classes, samples, words) = tokenize_corpus(traintxt, train=True)
        vocab = wordcount_filter(words, num=word_count_threshold)
        # Write new vocab file
        vocabf = outputf+"_vocab_"+str(word_count_threshold)+".txt"
        outfile = codecs.open(path+"/"+vocabf, 'w',"utf-8-sig")
        outfile.write("\n".join(vocab))
        outfile.close()
    else:
        word_count_threshold = 0
        (docs, classes, samples) = tokenize_corpus(traintxt, train=False)
        vocabfile = open(path+"/"+vocabf, 'r')
        vocab = [line.rstrip('\n') for line in vocabfile]
        vocabfile.close()

    print 'Vocabulary file:', path+"/"+vocabf 
    return (docs, classes, samples, vocab)

# fix this.
def main(argv):
    start_time = time.time()

    path = ''
    outputf = 'out'
    vocabf = ''

    try:
        opts, args = getopt.getopt(argv,"p:o:v:",["path=","ofile=","vocabfile="])
    except getopt.GetoptError:
        print 'Usage: \n python preprocess_text.py -p <path> -o <outputfile> -v <vocabulary>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'Usage: \n python preprocess_text.py -p <path> -o <outputfile> -v <vocabulary>'
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg
        elif opt in ("-o", "--ofile"):
            outputf = arg
        elif opt in ("-v", "--vocabfile"):
            vocabf = arg

    trainX = path+"/trainX_text.csv"
    trainY = path+"/trainY.csv"
    testX = path+"/testX_text.csv"
    testY = path+"/testY.csv"
    print 'Path:', path
    print 'Training data:', trainX, trainY
    print 'Testing data:', testX, testY

    train = construct_docs(trainX, trainY)
    test = construct_docs(testX, testY)
    (docs, classes, samples, vocab) = get_vocab(vocabf)

    # Get bag of words:
    bow = find_wordcounts(docs, vocab)
    # Check: sum over docs to check if any zero word counts
    print "Doc with smallest number of words in vocab has:", min(numpy.sum(bow, axis=1))

    # Write bow file
    with open(path+"/"+outputf+"_bag_of_words_"+str(word_count_threshold)+".csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(bow)

    # Process test.txt / write bow
    (docstest, classestest, samplestest) = tokenize_corpus(testtxt, train=False)
    testbow = find_wordcounts(docstest, vocab)
    with open(path+"/"+outputf+"_testX"+str(word_count_threshold)+".csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(testbow)

    # Write classes
    outfile= open(path+"/"+outputf+"_classes_"+str(word_count_threshold)+".txt", 'w')
    outfile.write("\n".join(classes))
    outfile.close()

    # Write test.txt classes
    outfile = open(path+"/"+outputf+"_testY"+str(word_count_threshold)+".txt", "w")
    outfile.write("\n".join(classestest))
    outfile.close()

    # Write samples
    outfile= open(path+"/"+outputf+"_samples_class_"+str(word_count_threshold)+".txt", 'w')
    outfile.write("\n".join(samples))
    outfile.close()

    print 'Output files:', path+"/"+outputf+"*"

    # Runtime
    print 'Runtime:', str(time.time() - start_time)

if __name__ == "__main__":
    main(sys.argv[1:])
