import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.read_data import read_data
from src.transform import Transformation


class Learner:

    def __init__(self):
        self.folder_ml = 'saved_models/'

    def train_model(self, X_train, y_train):

        """
        Training the model

        :param X_train: training features
        :type X_train: pandas dataframe
        :param y_train: training labels
        :type y_train: pandas dataframe

        """

        print('Fitting model...')
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        self.model.fit(X_train, y_train)

    def save_model(self, model_name):

        """
        Saving the model using pickle
        """

        filename = model_name + '.pkl'
        pickle.dump(self.model, open(self.folder_ml + filename, 'wb'))

    def load_model(self, model_name):

        """
        Loading the model
        """

        filename = model_name + '.pkl'
        self.model = pickle.load(open(self.folder_ml + filename, 'rb'))


class Evaluation:

    def __init__(self, use_saved_model=True):
        self.validation_pc = 0.2
        self.use_saved_model = use_saved_model
        self.X, self.y = read_data('data/train.csv', label_bool=True)
        self.transform = Transformation()
        self.learner = Learner()
        self.model_name = 'model_eval'

    def split_train_test(self, X, y):

        """
        A method to split the features into a train and test set

        :param X: features
        :type X: pandas dataframe
        :param y: labels
        :type y: pandas dataframe

        :returns:
            - X_train - train features
            - X_test - test features
            - y_train - train labels
            - y_test - test labels

        """

        # split the dataframes
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.validation_pc, random_state=42)

        return X_train, X_valid, y_train, y_valid

    def run(self):

        """
        Evaluate the performance of the model
        """

        # transform
        print('Tranforming data...')
        X_train, X_valid, y_train, y_valid = self.split_train_test(X=self.X, y=self.y)
        X_train, X_valid = self.transform.transform_data(X_train=X_train, X_test=X_valid)

        # train
        if self.use_saved_model:
            print('Loading model...')
            self.learner.load_model(model_name=self.model_name)
        else:
            print('Training model...')
            self.learner.train_model(X_train, y_train)
            self.learner.save_model(model_name=self.model_name)

        # evaluate
        print('Evaluating model...')
        y_valid_pred = self.learner.model.predict(X_valid)
        auc_score = roc_auc_score(y_valid, y_valid_pred)
        print('AUC score:', auc_score)


class Prediction:

    def __init__(self):
        self.X_train, self.y_train = read_data('data/train.csv', label_bool=True)
        self.X_test = read_data('data/test.csv', label_bool=False)
        self.transform = Transformation()
        self.learner = Learner()
        self.model_name = 'model_full'

    def train_model(self):

        """
        Training the model on training data
        """

        X_train, _ = self.transform.transform_data(X_train=self.X_train, X_test=self.X_train)
        self.learner.train_model(X_train, self.y_train)
        self.learner.save_model(model_name=self.model_name)

    def predict_test_values(self):

        """

        Making predictions on the test set

        :returns:
            - predictions - array of test set predictions

        """

        # transform data
        X_test, _ = self.transform.transform_data(self.X_test, self.X_test)

        # load model
        self.learner.load_model(model_name=self.model_name)

        # predict the probability of success
        predictions = self.learner.model.predict_proba(X_test)[:, -1][0]

        return predictions