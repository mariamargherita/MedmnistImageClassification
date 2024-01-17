
import pickle
import cv2
import numpy as np


def data_feed():
    """
    This function loads the data we extracted with data_import.
    :return: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Load data
    with open('data/X_train.pickle', 'rb') as f:
        X_train = pickle.load(f)

    with open('data/y_train.pickle', 'rb') as f:
        y_train = pickle.load(f)

    with open('data/X_val.pickle', 'rb') as f:
        X_val = pickle.load(f)

    with open('data/y_val.pickle', 'rb') as f:
        y_val = pickle.load(f)

    with open('data/X_test.pickle', 'rb') as f:
        X_test = pickle.load(f)

    with open('data/y_test.pickle', 'rb') as f:
        y_test = pickle.load(f)

    return X_train, y_train, X_val, y_val, X_test, y_test


def RGB2GRAY(X):
    """

    :param X:
    :return:
    """
    X_grey = []
    for i in range(len(X)):
        grey_img = cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY)
        X_grey.append(grey_img)

    X_grey = np.array(X_grey)

    return X_grey
