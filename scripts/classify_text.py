import numpy as np
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
        trainX = self.preprocess(trainX)
        print 'preprocessing test...'
        testX = self.preprocess(testX)
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
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
            precision_score(self.trainY, self.train_results[method_name]),
            recall_score(self.trainY, self.train_results[method_name]),
            f1_score(self.trainY, self.train_results[method_name]))
        print 'Test: precision=%.2f recall=%.2f f1=%.2f' % (
            precision_score(self.testY, self.test_results[method_name]),
            recall_score(self.testY, self.test_results[method_name]),
            f1_score(self.testY, self.test_results[method_name]))
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

    def plot_roc(self):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i, model in self.models.iteritems():
            fpr[i], tpr[i], _ = roc_curve(self.testY, self.test_results[i])
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
    Y = np.loadtxt(trainY_name)
    testX = np.loadtxt(testX_name, delimiter=",")
    testY = np.loadtxt(testY_name)
    runner = TrialRunner(X, Y, testX, testY, tfidf=tfidf, k=k)
    runner.run()
    # runner.plot_roc()
    runner.cross_val()

if __name__=='__main__':
    main("../data/out_courtlistener_train_bow5.csv", "../data/out_courtlistener_train_classes_5.txt", "../data/out_courtlistener_test_bow5.csv", "../data/out_courtlistener_test_classes_5.txt")
