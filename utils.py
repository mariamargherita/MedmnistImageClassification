
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def plot_accuracy_loss(history):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")

    axs[0].plot(history.history['loss'], '-o', label='Training')
    axs[0].plot(history.history['val_loss'], '-o', label='Validation')
    axs[1].plot(history.history['accuracy'], '-o', label="Training")
    axs[1].plot(history.history['val_accuracy'], '-o', label='Validation')

    axs[0].legend()
    axs[1].legend()
    plt.legend()
    plt.savefig("plots/cnn_model_performance.png")
