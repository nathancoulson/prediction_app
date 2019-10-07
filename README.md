# prediction_app
Prediciton and auto-scaling recommendation pipeline for MSc Data Science project repo (microservice auto-scaling system)

# Description
Application summary as part of auto-scaling prediciton and recommendation pipeline (seen in 2019 N C Coulson paper: "Adaptive microservice scaling for elastic web applications")

## Preliminaries
You should create empty folders called "Datasets", "Models" and "Results" in the root folder of the prediction app. The log text files should be kept in the parent folder of the prediction app called "Data".

## ETL pipeline

Our ETL pipeline takes NGINX log files in text format as input. It then extracts, cleans and transforms the data into pandas DataFrames which can be analysed and subjected to modelling.

The notebook 01-Auto-Scaler-ETL-Pipeline contains instructions and examples of all the high level ETL functions.

The functions are stored in the ETL_Pipeline_Module.

## Modelling

We have two modelling pipelines. One for the request mix prediction using Keras LSTM models and one for the supervised model which predicts the average request response time.

There are four modelling notebooks for single model training, grid search and stability testing.

In addition, the grid search folder contains scripts that can be run in the background for large grids.

The functions are stored in the Modelling_Module.

## Auto-scaling recommendations

Once the request mix and response time predictions models are trained and stored, you can use the auto-scaler-script to run on the latest set of logs and output scaling recommendations.

The functions are stored in the Auto_Scaler_Module.

# Deployment


