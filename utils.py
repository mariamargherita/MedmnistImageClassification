
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt


def data_feed(data_flag):
    """
    This function loads the data we extracted with data_import.
    :param data_flag: specifies which data we want to feed
    :return: X_train, y_train, X_val, y_val, X_test, y_test
    """
    if data_flag == "pathmnist":
        file_name1 = 'data/X_train.pickle'
        file_name2 = 'data/y_train.pickle'
        file_name3 = 'data/X_val.pickle'
        file_name4 = 'data/y_val.pickle'
        file_name5 = 'data/X_test.pickle'
        file_name6 = 'data/y_test.pickle'

    elif data_flag == 'breastmnist':
        file_name1 = 'data/X_train_b.pickle'
        file_name2 = 'data/y_train_b.pickle'
        file_name3 = 'data/X_val_b.pickle'
        file_name4 = 'data/y_val_b.pickle'
        file_name5 = 'data/X_test_b.pickle'
        file_name6 = 'data/y_test_b.pickle'

    else:
        raise ValueError("The data_flag should be either 'pathmnist' or 'breastmnist'.")

    # Load data
    with open(file_name1, 'rb') as f:
        X_train = pickle.load(f)

    with open(file_name2, 'rb') as f:
        y_train = pickle.load(f)

    with open(file_name3, 'rb') as f:
        X_val = pickle.load(f)

    with open(file_name4, 'rb') as f:
        y_val = pickle.load(f)

    with open(file_name5, 'rb') as f:
        X_test = pickle.load(f)

    with open(file_name6, 'rb') as f:
        y_test = pickle.load(f)

    return X_train, y_train, X_val, y_val, X_test, y_test


def RGB2GRAY(X):
    """
    Converts RGB images to gray images.
    :param X: dataset of images
    :return: same dataset of images but in grey scale
    """
    X_grey = []
    for i in range(len(X)):
        grey_img = cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY)
        X_grey.append(grey_img)

    X_grey = np.array(X_grey)

    return X_grey


def plot_accuracy_loss(history):
    """
    This function plots model accuracy vs. validation accuracy and loss vs. validation loss.
    :param history: train history
    :return: acc vs. val_acc and loss vs. val_loss plots
    """
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


def accuracy_fn(y_true, y_pred):
    """
    This function computes model accuracy.
    :param y_true: actual target
    :param y_pred: predicted target
    :return: model accuracy
    """

    boolean_result = []
    for t, p in zip(y_true, y_pred):
        if t == p:
            boolean_result.append(True)
        else:
            boolean_result.append(False)

    correct = np.array(boolean_result).sum()
    accuracy = correct / len(y_pred)
    return accuracy
