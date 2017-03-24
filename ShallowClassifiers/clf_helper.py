'''

2) General Functions for classifiers. Runs the classifiers and prints the result

'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

#For classificaition
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron

from time import time

''' Benchmark classifiers'''
def benchmark(clf, X_train,X_test,y_train,y_test):

    from sklearn.model_selection import cross_val_predict
    

    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    #pred = cross_val_predict(clf, X_test,y_test, cv=10)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


''' Add or remove classifier to pipline ''' 
def classify(X_train,X_test,y_train,y_test):

    results = []
    for clf, name, do_run in (
            (Perceptron(n_iter=50), "Perceptron",False),
            (LinearSVC(), "Linear Support Vector Machine",True)):

        if (do_run==True):
            print('=' * 80)
            print(name)
            clf_descr, score, train_time, test_time = benchmark(clf,X_train,X_test,y_train,y_test)
            results.append([clf_descr, score, train_time, test_time])

    return results,score