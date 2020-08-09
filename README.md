# Machine Learning Template

This repository helps to solve two problems in local model development for data scientists:
* A lot of time is often spend in setting up the coding framework
to run your machine learning model (e.g. splitting train/test data,
training/evaluating model etc). This repo sets an object-orientated,
easy-to-understand code base which can be adjusted accordingly
to your specific problem. 
*  Versioning your model and tracking your model performance with different 
inputs can be challenging. This repo uses [dvc](https://dvc.org/) 
(to version the data) and [mlflow](https://mlflow.org/docs/latest/tracking.html)
(to track model performance).
 

## Getting started

To try out the package, follow the steps below:

* Clone this repository to local machine
* cd in folder root
* `chmod +x setup.sh` to make bash file executable
* `./setup.sh` to run executable - this installs a virtualenv and downloads relevant data

Then add your data!

## Starting to use dvc

[DVC](https://dvc.org/) stands for 'data version control' - it is the git for data. To start using dvc,
follow the below steps (for advanced usage, see the website):

1. cd to folder root
2. `source venv/bin/activate` - activating virtual environment
3. `dvc init` - creates a '.dvc' directory to initial the dvc package
4. `dvc add data/train.csv` - creates a .dvc file (a placeholder for the original data), a small text file
in a human-readable format
5. `git add data/train.csv.dvc & git commit -m "Add raw data"` - this adds the data to git

Whenever the data changes, simply run the `dvc add [file]` command and commit to save into version control.

## Starting to use mlflow

[MLFlow](https://mlflow.org/docs/latest/tracking.html) helps to track and log machine learning experiments. 
To start using mlflow, follow the below steps:

1. Adjust the code as necessary to fit your data/problem (e.g. you might need to do some data transformation / want to
use a different ML model / use a different metric)
2. `mlflow ui` into the root directory of the project
3. In `main.py`, select the evaluation method (`run_single` or `run_cv`)
4. When ready to record a run, set `mlflow_record=True` and enter an experiment name (each experiment can have 
multiple runs associated with it)
5. Navigate to the UI (localhost:5000) and see the run recorded

Here are some top tips for tracking your model using MLflow, based on [this](https://medium.com/ixorthink/our-machine-learning-workflow-dvc-mlflow-and-training-in-docker-containers-5b9c80cdf804) blog post:

* If you are debugging, then set the variable `ml_flow_record` to false. This ensures that you only
record the runs which are significant, aiding model analysis
* Before you record a run in MLFlow, make sure you commit the code (via Git) and the
data (via dvc). This ensures you have an accurate record in MLFlow on what code and data
produced the model.
* When experimenting with new data / new approach, create a fresh branch in git. This uses the power of this framework to make it very easy to compare / switch back to existing setup
* Use experiments in MLFlow to group together runs

## Files

This repo contains the following files:

* `src/read_data.py` - files which reads in the data. The code currently works for data
in a format detailed in `data/data_structure.txt`
* `src/transform.py` - a file for all your necessary data transformations to get the data
into a form that can be consumed by ML model. This might involve
filling in missing data, encoding categorical variables etc
* `src/ml.py` - a file that contains the machine learning code, including classes which
train the model, evaluate the model and predict on the test data
* `main.py` - file from which to run the source code

## Contributing

Please do contribute to improve the repository. If you have an issue with the current code/documentation, do open an issue
[here](https://github.com/eddiepease/ml-template/issues)