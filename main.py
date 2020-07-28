""" Module to run src files """

from src.ml import Evaluation, Prediction


def run():

    """

    Function to run the machine learning model

    """

    eval_model = Evaluation(mlflow_record=False)
    eval_model.run_single(exp_name='single_run')

    # predict = Prediction()
    # predict.train_model()


if __name__ == '__main__':

    run()
