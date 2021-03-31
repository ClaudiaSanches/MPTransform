"""
    Example of code that computes de MPDist for different values for the length of subsequences,
    creates the dataset of distances, trains, test and evaluate the results.
"""

import numpy as np
import pandas as pd

from base_datasets import _load_dataset
from mpdist import mpdist

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import logging
import time


def create_train_dataset(X_train, k):
    """
        Function that calculates the MP distance between every possible
        pair of series in the training portion of the dataset, storing
        the results in a .csv file.

        Parameters
        ----------
            X_train: numpy.array
                Time series from the train set.
            k: float
                Percentage of each series total length used to determine
                the value for m (the subsequence length).

    """

    n = int(k*100)
    start = time.time()

    m = int(k * X_train[0].shape[0]) # determine the value for m according to k
    temp = np.zeros(shape=(len(X_train), len(X_train)))

    for i in range(0, len(X_train)):
        for j in range(i, len(X_train)):
            temp[i][j] = temp[j][i] = mpdist(X_train[i], X_train[j], m)
    # Here, it is only necessary to calculate the values for the top (or bottom)
    # diagonal, since the distance between the ith and the jth series is the same
    # as between the jth and ith

    df = pd.DataFrame(temp)
    df.to_csv('datasets/Car/car_train_'+str(n)+'.csv', index=False)

    logging.info('Dataset car_train_%d created!', n)
    end = time.time()
    logging.info('It took %f to create dataset car_train_%d\n', end-start, n)


def create_test_dataset(X_train, X_test, k):
    """
        Function that calculates the MP distance between each series in
        the testing set and every other series in the training set, storing
        the results in a .csv file.

        Parameters
        ----------
            X_train: numpy.array
                Time series from the train set.
            X_test: numpy.array
                Time series from the test set.
            k: float
                Percentage of each series total length used to determine
                the value for m (the subsequence length).

    """

    n = int(k*100)
    start = time.time()

    m = int(k * X_train[0].shape[0]) # determine the value for m according to k
    temp = np.zeros(shape=(len(X_test), len(X_train)))

    for i in range(0, len(X_test)):
        for j in range(0, len(X_train)):
             temp[i][j] = mpdist(X_test[i], X_train[j], m)

    df = pd.DataFrame(temp)
    df.to_csv('datasets/Car/car_test_'+str(n)+'.csv', index=False)

    logging.info('Dataset car_test_%d created!', n)
    end = time.time()
    logging.info('It took %f to create dataset car_test_%d\n', end-start, n)


X_train, y_train = _load_dataset('Car', split='TRAIN', return_X_y=True)
X_test, y_test = _load_dataset('Car', split='TEST', return_X_y=True)

X_train = X_train.to_numpy()
X_train = X_train.flatten()
X_test = X_test.to_numpy()
X_test = X_test.flatten()

logging.basicConfig(filename='car.log', filemode='w', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

svm_acc = 0
rf_acc = 0
bayes_acc = 0

for i in range(9, 1, -1):
    # Create .csv file for the training data
    logging.info('Creating dataset car_train_%d...', i*10)
    create_train_dataset(X_train, i/10)

    # Create .csv file for the testing data
    logging.info('Creating dataset car_test_%d...', i*10)
    create_test_dataset(X_train, X_test, i/10)

    # Training
    x_train = pd.read_csv("datasets/Car/car_train_" + str(i*10) + ".csv")

    svm_clf = svm.SVC(max_iter=1000, gamma='auto')
    rf_clf = RandomForestClassifier(random_state=0)
    bayes_clf = MultinomialNB()

    svm_model = svm_clf.fit(x_train, y_train)
    rf_model = rf_clf.fit(x_train, y_train)
    bayes_model = bayes_clf.fit(x_train, y_train)

    # Testing
    x_test = pd.read_csv("datasets/Car/car_test_" + str(i*10) + ".csv")

    svm_pred = svm_clf.predict(x_test)
    rf_pred = rf_clf.predict(x_test)
    bayes_pred = bayes_clf.predict(x_test)

    logging.info('m = %d', i*10)
    logging.info('SVM accuracy: %f', accuracy_score(y_test, svm_pred))
    logging.info('Random Forest accuracy: %f', accuracy_score(y_test, rf_pred))
    logging.info('Naive Bayes accuracy: %f\n', accuracy_score(y_test, bayes_pred))

    if (accuracy_score(y_test, svm_pred) > svm_acc):
        svm_acc = accuracy_score(y_test, svm_pred)

    if (accuracy_score(y_test, rf_pred) > rf_acc):
        rf_acc = accuracy_score(y_test, rf_pred)

    if (accuracy_score(y_test, bayes_pred) > bayes_acc):
        bayes_acc = accuracy_score(y_test, bayes_pred)

logging.info('----- BEST ACCURACIES SO FAR -----')
logging.info('SVM: %f', svm_acc)
logging.info('Random Forest: %f', rf_acc)
logging.info('Multinomial Naive Bayes: %f\n', bayes_acc)

logging.info('*** End of compilation ***')
