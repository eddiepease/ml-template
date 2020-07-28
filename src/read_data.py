""" Module to read data """

import pandas as pd


def read_data(file_path, label_bool):

    """

    Reads the data into a pandas dataframe. This assumes that features and labels are in same CSV

    :param file_path: path to data
    :type file_path: str
    :param label_bool: whether the label is included in dataframe
    :type label_bool: bool

    :returns:
        - X - features dataframe
        - y - labels dataframe

    """

    # read in dataframe
    data = pd.read_csv(file_path)

    # split into features and labels
    if label_bool:
        label_column = 'label'
        X = data.drop(labels=label_column, axis=1)
        y = data[label_column]
        return X, y
    else:
        X = data
        return X
