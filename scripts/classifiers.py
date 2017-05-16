#!/usr/local/bin/python

import csv
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import re
from time import time
from optparse import OptionParser
#np.set_printoptions(threshold=np.nan)

from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer



# TODO integrate better data structures
def load_file(file):
	vect = []
	with open(file, 'r') as f: 
		reader = csv.reader(f)
		for row in reader:
			vect.append(row) # store vals as ints (not char)
	vect = np.array(vect)
	return vect

def load_file_txt(file):
	vect = []
	f = open(file, 'r')
	for row in f.readlines():
		vect.append(int(row))
	vect = np.array(vect)
	return vect

def rmseContinuous(clf, train_data, train_class, test_data, test_class):
	clf.fit(train_data, train_class)
	predicted = clf.predict(test_data)
	print "RMSE: ", mean_squared_error(test_class, predicted)
	print "R^2: ", r2_score(test_class, predicted)

def predictFit(clf, train_data, train_class, test_data, test_class):

	clf.fit(train_data, train_class)
	predicted = clf.predict(test_data)
	# return
	print "score: ", metrics.accuracy_score(test_class, predicted) # = np.mean(predicted == test_class)
	print "confusion matrix:"
	print metrics.confusion_matrix(test_class, predicted)
	# print "f1 score: ", metrics.f1_score(test_class, predicted)
	print metrics.classification_report(test_class, predicted)
 	#print metrics.classification_report(train_class, predicted)


# p


def classify():
	# TODO: make files command line args
	train_bag = pd.read_csv('../data/trainX_num_output.csv')
#	print train_bag.shape
	temp_train_class = pd.read_csv('../data/trainY.csv')
	train_class = temp_train_class.pop('partyWinning')
#	print train_class.shape
	test_bag = pd.read_csv('../data/testX_num_output.csv')
	# print test_bag.shape
#	print test_bag.shape
	temp_test_class = pd.read_csv('../data/testY.csv')
	test_class = temp_test_class.pop('partyWinning')
#	print test_class.shape

	label = "Extended"
	print "Naive Bayes"
	start = time.time()
	naive = MultinomialNB()
	predictFit(naive, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print(end - start)
	predict_probas_nb = naive.predict_proba(test_bag)[:,1]
	fpr_nb, tpr_nb, _ = metrics.roc_curve(test_class, predict_probas_nb)
	roc_auc_nb = metrics.auc(fpr_nb, tpr_nb)
	# lb = 'BOW-NB' + label
	# plot_ROC(fpr_nb, tpr_nb, roc_auc_nb, 'NB' + label)
	print

	print "SVM (Gaussian Kernel)"
	# svm = SGDClassifier(loss='log', penalty='l2',
	# 	alpha=1e-3, n_iter=5, random_state=42)
	start = time.time()
	svmg = SVC(kernel='rbf', probability=True)
	predictFit(svmg, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print(end - start)
	predict_probas_svmg = svmg.predict_proba(test_bag)[:,1]
	fpr_svmg, tpr_svmg, _ = metrics.roc_curve(test_class, predict_probas_svmg)
	roc_auc_svmg = metrics.auc(fpr_svmg, tpr_svmg)
	# plot_ROC(fpr_svmg, tpr_svmg, roc_auc_svmg, 'SVMG' + label)
	print

	print "Logistic Regression"
	start = time.time()
	logreg = linear_model.LogisticRegression()
	predictFit(logreg, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print(end - start)
	predict_probas_lr = logreg.predict_proba(test_bag)[:,1]
	fpr_lr, tpr_lr, _ = metrics.roc_curve(test_class, predict_probas_lr)
	roc_auc_lr = metrics.auc(fpr_lr, tpr_lr)
	# plot_ROC(fpr_lr, tpr_lr, roc_auc_lr, 'LR' + label)

	print "Decision Tree"
	#dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
	start = time.time()
	dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
	predictFit(dt, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print(end - start)
	predict_probas_dt = dt.predict_proba(test_bag)[:,1]
	fpr_dt, tpr_dt, _ = metrics.roc_curve(test_class, predict_probas_dt)
	roc_auc_dt = metrics.auc(fpr_dt, tpr_dt)
	return

def continuousClassifiers(alpha):
	print "Alpha: "
	print alpha
	train_bag = pd.read_csv('../data/trainX_num_output.csv')
#	print train_bag.shape
	temp_train_class = pd.read_csv('../data/trainY.csv')
	train_class = temp_train_class.pop('partyWinning')
#	print train_class.shape
	test_bag = pd.read_csv('../data/testX_num_output.csv')
	# print test_bag.shape
#	print test_bag.shape
	temp_test_class = pd.read_csv('../data/testY.csv')
	test_class = temp_test_class.pop('partyWinning')

	print "Random Forest"
	rf = RandomForestRegressor()
	rmseContinuous(rf, train_bag, train_class, test_bag, test_class)

	print "Elastic Net"
	enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
	rmseContinuous(enet, train_bag, train_class, test_bag, test_class)

	print "Lasso"
	lasso = Lasso(alpha=alpha)
	rmseContinuous(lasso, train_bag, train_class, test_bag, test_class)

	print "Ridge"
	ridge = Ridge(alpha=alpha,normalize=True)
	rmseContinuous(ridge, train_bag, train_class, test_bag, test_class)


def features():
	# split a training set and a test set
	# y_train, y_test = data_train.target, data_test.target

	train_bag = pd.read_csv('../data/trainX_num_output.csv')
	print train_bag.shape
	temp_train_class = pd.read_csv('../data/trainY.csv')
	y_train = temp_train_class.pop('partyWinning')
#	print train_class.shape
	test_bag = pd.read_csv('../data/testX_num_output.csv')
	# print test_bag.shape
#	print test_bag.shape
	temp_test_class = pd.read_csv('../data/testY.csv')
	y_test = temp_test_class.pop('partyWinning')

	print("Extracting features from the training data using a sparse vectorizer")
	t0 = time()
	# if opts.use_hashing:
	#     vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
	#                                    n_features=opts.n_features)
	#     X_train = vectorizer.transform(train_bag)
	# else:
	#     vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
	#                                  stop_words='english')
	#     X_train = vectorizer.fit_transform(train_bag)
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
	                                 stop_words='english')
	X_train = vectorizer.fit_transform(train_bag)
	duration = time() - t0
	# print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
	print("n_samples: %d, n_features: %d" % X_train.shape)
	print()

	print("Extracting features from the test data using the same vectorizer")
	t0 = time()
	X_test = vectorizer.transform(test_bag)
	duration = time() - t0
	# print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
	print("n_samples: %d, n_features: %d" % X_test.shape)
	print()

	feature_names = vectorizer.get_feature_names()

	# if opts.select_chi2:
	#     print("Extracting %d best features by a chi-squared test" %
	#           opts.select_chi2)
	t0 = time()
	ch2 = SelectKBest(chi2, k=10)
	X_train = ch2.fit_transform(X_train, y_train)
	X_test = ch2.transform(X_test)
	feature_names = [feature_names[i] for i
	                         in ch2.get_support(indices=True)]
	print("done in %fs" % (time() - t0))
	print()

	feature_names = np.asarray(feature_names)

	print feature_names

def main():
	classify()
	continuousClassifiers(.01)
	continuousClassifiers(.1)
	continuousClassifiers(.5)
	continuousClassifiers(1)
	features()

main()
