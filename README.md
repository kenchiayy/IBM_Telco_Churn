# IBM_Telco_Churn
Predict whether or not a customer will stop using a company's service called Customer Churn. XGBoost model will use as prediction. *XGBoost* is a collection of boosted trees and it is an exceptionally useful machine learning method when you don't want to sacrifice the ability to correct classify observations but you still want a model that is fairly easy to understand and interpret.


### Requirements
These are the requirements:
- jupyter==1.0.0
- jupyter-client==6.0.0
- jupyter-core==4.6.3
- jupyter_console==6.0.0
- jupyterlab==1.0.2
- jupyterlab_server==1.0.0
- notebook==6.0.3
- qtconsole==4.5.1
- ipykernel==5.5.0
- ipython==7.6.1
- nbconvert==5.5.0
- ipywidgets==7.5.0
- nbformat==4.4.0
- traitlets==4.3.2
- numpy==1.20.1
- pandas==1.2.2
- matplotlib==3.3.1
- seaborn==0.10.1
- xgboost==1.2.0
- scipy==1.5.2
- sklearn==0.23.2
- feature_engine==1.0.0
- pickle==4.0
- virtualenv==20.4.2
- pytest==5.0.1


### Training

Path: Train

The training of the model will be done in Jupyter Notebook, it will consist of the the following steps:

- Importing Data from a File
- Exploratory Data Analysis
- Missing Data
- Indentifying Missing Data
- Dealing with Missing Data
- Formatting the Data for XGBoost
  - Splitting data into Dependent and Independent Variables
  - Ordinal Label Encoding (Monotonic relationship)
  - Converting all columns to Int, Float or Bool
Building a Prelimary XGBoost Model

Optimizing Parameters with Cross Validation and GridSearch

Optimizing the learning rate, tree depth, number of trees, gamma (for prunning) and lambda (for regularization).
Deploy model using pickle

### Testing

Path: Test

Here will load the trained model and test it using pytest
