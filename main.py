from src.ml import Evaluation, Prediction


def run():

    """

    Function to run the machine learning model

    """

    eval = Evaluation(mlflow_record=False)
    eval.run()

    # predict = Prediction()
    # predict.train_model()


if __name__ == '__main__':

    run()