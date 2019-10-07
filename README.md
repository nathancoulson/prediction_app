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

We deployed the microservice based web application on a 2 Core, 4GB ram Ubuntu VM using Docker Swarm
We deployed the Locust.io container, which generates the simulated traffic, on an identical VM
We deployed the machine learning pipeline on another VM (32 cores, 128GB ram) to prevent interference due to resource utilisation

During the experiment runs we used a linux utility called cronjobs to synchronise the NGINX log files from the microservice app VM with the machine learning pipeline VM every minute and to run the pipeline on the most recent 1000 lines of log data.

This is the cronjob we used on the microservice application VM:

\* \* \* \* \* rsync -ratlz --rsh="/usr/bin/sshpass -p 1234 ssh -o StrictHostKeyChecking=no -l nathan" /home/nathan/app_manager/experiment-1.log nathan@193.61.29.193:/home/nathan/Data

This is the cronjob we used on the auto-scaler system VM:

\*/2 \* \* \* \* /home/nathan/prediction_app/env/bin/python /home/nathan/auto-scaler-script.py 1 3 1 1 ./Data/experiment-1.log ./Models/ >> /home/nathan/cron-output.log 2>&1

In experimental runs in which the scaling intervention was permitted we implemented this using the Docker API by scaling the recommended microservice manually through the command line interface.
