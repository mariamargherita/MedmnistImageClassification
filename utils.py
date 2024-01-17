
import pickle


def data_feed():
    """
    This function loads the data we extracted with data_import.
    :return: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Load data
    with open('X_train.pickle', 'rb') as f:
        X_train = pickle.load(f)

    with open('y_train.pickle', 'rb') as f:
        y_train = pickle.load(f)

    with open('X_val.pickle', 'rb') as f:
        X_val = pickle.load(f)

    with open('y_val.pickle', 'rb') as f:
        y_val = pickle.load(f)

    with open('X_test.pickle', 'rb') as f:
        X_test = pickle.load(f)

    with open('y_test.pickle', 'rb') as f:
        y_test = pickle.load(f)

    return X_train, y_train, X_val, y_val, X_test, y_test
