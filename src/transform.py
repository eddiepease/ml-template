import pandas

from src.read_data import read_data


class Transformation:

    def __init__(self):
        pass

    def transform_data(self, X_train, X_test):

        """
        A method to perform necessary transformations

        :param X_train: training data
        :type X_train: pandas dataframe
        :param X_test: test data
        :type X_test: pandas dataframe

        :returns:
            - X_train - transformed training data
            - X_test - transformed test data

        """

        X_train = X_train
        X_test = X_test

        return X_train, X_test