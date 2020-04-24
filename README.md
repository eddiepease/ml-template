# ml-workflow

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

Then add your data and adjust the code as required!

## Features

This repo contains the following files:

* `src/read_data.py` - files which reads in the data. The code currently works for data
in a format detailed in `data/data_structure.txt`
* `src/transform.py` - a file for all your necessary data transformations to get the data
into a form that can be consumed by ML model. This might involve
filling in missing data, encoding categorical variables etc
* `src/ml.py` - a file that contains the machine learning code, including classes which
train the model, evaluate the model and predict on the test data
* `main.py` - file from which to run the source code

Here are some top tips for tracking your model using MLflow, based on [this](https://medium.com/ixorthink/our-machine-learning-workflow-dvc-mlflow-and-training-in-docker-containers-5b9c80cdf804) blog post:

* If you are debugging, then set the variable `ml_flow_record` to false. This ensures that you only
record the runs which are significant, aiding model analysis
* Before you record a run in MLFlow, make sure you commit the code (via Git) and the
data (via dvc). This ensures you have an accurate record in MLFlow on what code and data
produced the model.

## Contributing

Please do contribute to improve the repository. If you have an issue with the current code/documentation, do open an issue
[here](https://github.com/eddiepease/ml-workflow/issues)