import numpy as np
import pandas as pd
from sklearn import svm, naive_bayes
from sklearn.feature_extraction.text import TfidfTransformer
from time import time
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import binarize
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt


SCORE = 'macro' # average kwarg for precision, recall, f1
#SCORE = 'weighted'

def filter_out_empty(trainX, trainY, testX, testY):
    print trainX.shape, trainY.shape, testX.shape, testY.shape
    train = np.hstack((trainX, trainY.reshape(trainY.shape[0], 1)))
    test = np.hstack((testX, testY.reshape(testY.shape[0], 1)))
    x_cols = trainX.shape[1]
    train = train[trainY > 0]
    trainX = train[:,:x_cols]
    trainY = train[:,x_cols:]
    test = test[testY > 0]
    testX = test[:,:x_cols]
    testY = test[:,x_cols:]
    return trainX, trainY.ravel(), testX, testY.ravel()

class TrialRunner():
    models = {}
    train_results = {}
    test_results = {}

    def __init__(self, trainX, trainY, testX, testY, binary=False,
                 tfidf=True, k=None):
        self.methods = [#(svm.SVC(kernel='rbf', **{'C': 1, 'gamma': 0.5}),
                        # 'SVM-RBF'),
                        (svm.SVC(kernel='rbf'), 'SVM-RBF0'),
                        #(svm.SVC(kernel='linear', C=0.25), 'SVM-Linear'),
                        (svm.SVC(kernel='linear'), 'SVM-L0'),
                        (naive_bayes.GaussianNB(), 'GaussianNB()'),
                        (naive_bayes.MultinomialNB(), 'MultinomialNB()'),
                        (RandomForestClassifier(**{'max_depth': 40,
                                                   'min_samples_split': 12,
                                                   'n_estimators': 10}),
                         'RandomForest'),
                        #(RandomForestClassifier(n_estimators=50), 'RF50')
        ]
        self.binary = binary
        self.k = k
        self.tfidf = tfidf
        print 'preprocessing train...'
        trainX = self.preprocess(np.array(trainX, dtype=float))
        print 'preprocessing test...'
        testX = self.preprocess(np.array(testX, dtype=float))
        self.trainX = trainX
        self.trainY = np.array(trainY, dtype=float)
        self.testX = testX
        self.testY = np.array(testY, dtype=float)
        if k is not None:
            print 'selecting features...'
            self.select_features()
        self.fullX = np.vstack([self.trainX, self.testX])
        self.fullY = np.concatenate([self.trainY, self.testY])

        if binary:
            self.methods.append((naive_bayes.BernoulliNB(), 'BernouilliNB()'))

    def select_features(self):
        selector = SelectKBest(score_func=chi2, k=self.k)
        selector.fit(self.trainX, self.trainY)
        self.trainX = selector.transform(self.trainX)
        self.testX = selector.transform(self.testX)

    def preprocess(self, X):
        print X.shape, X.dtype
        if self.tfidf:
            transformer = TfidfTransformer(smooth_idf=False)
            X2 = transformer.fit_transform(X).toarray()
            threshold = X2.mean()
        else:
            X2 = X
            threshold = 0.5
            # if not using tfidf, then binary value represents presence
        if self.binary:
            return binarize(X2, threshold=threshold)
        else:
            return X2

    def add_method(self, clf, params, name):
        '''
        clf: a classifier class
        params: dict of kwargs to clf with parameter values e.g. {max_depth: [25, 50]}
        '''
        gs = GridSearchCV(clf(), params)
        gs.fit(self.fullX, self.fullY, scoring='f1_macro', cv=5)
        best_params = gs.best_params_
        print 'best params for %s: %s' % (name, best_params)
        self.methods.append((clf(**best_params), name))

    def print_result(self, method, method_name):
        print 'Train: precision=%.2f recall=%.2f f1=%.2f' % (
            precision_score(self.trainY, self.train_results[method_name],
                            average=SCORE),
            recall_score(self.trainY, self.train_results[method_name],
                         average=SCORE),
            f1_score(self.trainY, self.train_results[method_name],
                     average=SCORE)
        )
        print 'Test: precision=%.2f recall=%.2f f1=%.2f' % (
            precision_score(self.testY, self.test_results[method_name],
                            average=SCORE),
            recall_score(self.testY, self.test_results[method_name],
                         average=SCORE),
            f1_score(self.testY, self.test_results[method_name],
                     average=SCORE)
        )
        print '\n'

    def run(self):
        print 'Running classifiers...'
        for method, method_name in self.methods:
            print '%s -- ' % method_name
            start = time()
            model = method.fit(self.trainX, self.trainY)
            fit = time()
            self.models[method_name] = model
            self.train_results[method_name] = model.predict(self.trainX)
            self.test_results[method_name] = model.predict(self.testX)
            print ('time to fit: %.2fs, time to predict: %.2fs, '
                   'total time: %.2fs'
                   % (fit - start, time()-fit, time()-start))
            self.print_result(method, method_name)

    def plot_roc(self, pos_label=None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i, model in self.models.iteritems():
            fpr[i], tpr[i], _ = roc_curve(self.testY, self.test_results[i],
                                          pos_label=pos_label)
            roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure()
        lw = 2
        for i, name in enumerate(fpr.iterkeys()):
            plt.plot(fpr[name], tpr[name], color='C%s' %i,
                     lw=lw, label='%s (area = %0.2f)' % (name, roc_auc[name]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()

    def cross_val(self, k=5):
        print '%s fold Cross Validation' % k
        for i, model in self.models.iteritems():
            start = time()
            scores = cross_val_score(model, self.fullX, self.fullY, cv=k,
                                     scoring='f1_macro')
            print("%s Accuracy: %0.2f (+/- %0.2f), Time:%0.2fs" %
                  (i, scores.mean(), scores.std() * 2, time()-start))

"""
def get_accuracy(Y, Z):
    correct = np.zeros(Y.shape)
    for i, (y, z) in enumerate(zip(Y, Z)):
        if y == z:
            correct[i] = 1
    return correct.mean()
"""

def main(trainX_name, trainY_name, testX_name, testY_name, tfidf=True, k=None):
    # load training and test sets
    X = np.loadtxt(trainX_name, delimiter=',')
    #Y = np.loadtxt(trainY_name)
    testX = np.loadtxt(testX_name, delimiter=",")
    #testY = np.loadtxt(testY_name)

    # copied from wordclouds.py
    voteIds = np.loadtxt('../data/out_courtlistener_all_samples.txt',
                         dtype=str)
    voteIds = voteIds.reshape(voteIds.shape[0], 1)
    scdb = pd.read_csv('../data/SCDB_2016_01_justiceCentered_Citation.csv')
    scdb = scdb.fillna(value=-1)
    samples = pd.DataFrame(data=voteIds, columns=['voteId'])
    samples = samples.merge(scdb, how='left', on='voteId')
    train_len = X.shape[0]

    pos_label = 2 # liberal, or majority
    for label in ['direction', 'majority']: # these are binary, vote is not.
        print 'label:', label
        Y = np.array(samples[label].values, dtype=float)
        trainY = Y[:train_len]
        testY = Y[train_len:]
        this_X, trainY, this_testX, testY = filter_out_empty(X, trainY, testX, testY)
        runner = TrialRunner(this_X, trainY, this_testX, testY, tfidf=tfidf, k=k)
        runner.run()
        runner.plot_roc(pos_label=pos_label)
        # runner.cross_val()

def run_by_direction(tfidf=True, k=None):
    # conservative/liberal
    import wordclouds as wc
    data = wc.get_data()
    conservative = wc.get_conservative(data)
    liberal = wc.get_liberal(data)
    pos_label = 2 # majority
    for dataset in [conservative, liberal]:
        if dataset == conservative:
            print 'conservative'
        else:
            print 'liberal'
        bow, votes, samples, vocab = dataset
        X = np.delete(bow, 0, axis=1)
        Y = np.array(samples.majority.values, dtype=float)
        trainX, trainY, testX, testY = wc.split_train_test(X, Y)
        runner = TrialRunner(trainX, trainY, testX, testY, tfidf=tfidf, k=k)
        runner.run()
        runner.plot_roc(pos_label=pos_label)

def find_author(tfidf=True, k=None):
    # import
    # run classifier to see how well we can predict the author
    pass

if __name__=='__main__':
    main("../data/out_courtlistener_bow5362_1000.csv", "../data/out_courtlistener_classes5362_1000.txt", "../data/out_courtlistener_test_bow5362_1000.csv", "../data/out_courtlistener_testY5362_1000.txt", k=100)
    run_by_direction(k=100)
