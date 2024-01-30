
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers
import tensorflow
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


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


def augment_data(X, y, n_augs=9):
    """
    Performs data augmentation on provided dataset.
    :param X: predictors set
    :param y: target set
    :param n_augs: number of times we want to augment an image
    :return: augmented dataset
    """
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=12),
        layers.RandomRotation(0.2, seed=23)
    ])

    X_aug = []
    y_aug = []
    for index in tqdm(range(len(X))):
        image = X[index]
        image = image.reshape(image.shape + (1,))
        label = y[index]
        for i in range(n_augs):
            augmented_image = data_augmentation(tensorflow.convert_to_tensor(image))
            augmented_image = augmented_image.numpy()
            augmented_image = augmented_image.reshape(augmented_image.shape[:-1])
            X_aug.append(augmented_image)
            y_aug.append(label)

    X_aug = np.asarray(X_aug)
    y_aug = np.asarray(y_aug)

    shuffle_index = np.random.permutation(len(X_aug))
    X_aug = X_aug[shuffle_index]
    y_aug = y_aug[shuffle_index]

    return X_aug, y_aug


def fit_grid(X, y, param_combination):
    """

    :param grid_search:
    :param X:
    :param y:
    :param param_combination:
    :return:
    """

    classifier = RandomForestClassifier(**param_combination, verbose=1)
    sk_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)
    scores = cross_val_score(classifier, X, y, cv=sk_fold)
    cv_score = np.mean(scores)

    print(f"CV score: {cv_score}")

    return cv_score, param_combination


'''
def plot_grid(grid_search, x, y):
    x_axis =
    y_axis =
    scores = [i[1] for i in grid_search.grid_scores_]
    scores = np.array(scores).reshape(len(Cs), len(Gammas))

    for ind, i in enumerate(Cs):
        plt.plot(Gammas, scores[ind], label='C: ' + str(i))
    plt.legend()
    plt.xlabel('Gamma')
    plt.ylabel('Mean score')
    plt.show()
'''