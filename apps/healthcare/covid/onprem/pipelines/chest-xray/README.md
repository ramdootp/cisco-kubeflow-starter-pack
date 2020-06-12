# Chest Xray Predicton using Pipelines

## What we're going to build

Train and serve Chest Xray model using KF pipeline, and predict xray result from Jupyter notebook.

![TF-Chest Xray Pipeline](pictures/0-xray-graph.PNG)

## Infrastructure Used

* Cisco UCS - C240M5 and C480ML

## Setup

### Install NFS server (if not installed)

To install NFS server follow [steps](../../../../../networking/ble-localization/onprem/pipelines#install-nfs-server-if-not-installed)

### Create Jupyter Notebook Server

Follow the [steps](./../../notebook#create--connect-to-jupyter-notebook-server)  to create Jupyter Notebook in Kubeflow

### Upload Notebook file to generate dataset

Upload Chest-Xray-Dataset-Builder.ipynb file from [here](./Chest-Xray-Dataset-Builder.ipynb)

### Upload Notebook file for pipeline

Upload Chest-Xray-Pipeline-Deployment.ipynb file from [here](./Chest-Xray-Pipeline-Deployment.ipynb)

### Run Chest Xray Pipeline

Open the Chest-Xray-Pipeline-Deployment.ipynb file and run pipeline

Clone git repo

![TF-Chest Xray  Pipeline](pictures/1-git-clone.png)

Loading Components

![TF-BLERSSI Pipeline](pictures/2-load-compoents.PNG)

Run Pipeline

![TF-Chest Xray Pipeline](pictures/2-run-pipeline.PNG)

Once Chest Xray Pipeline is executed Experiment and Run link will generate and displayed as output

![TF-Chest Xray Pipeline](pictures/3-exp-link.PNG)

Click on latest experiment which is created

![TF-Chest Xray Pipeline](pictures/4-pipeline-created.PNG)

Pipeline components execution can be viewed as below
Logs of Chest Xray Training Component
![TF-Chest Xray Pipeline](pictures/6-pipeline-completed.PNG)

Logs of Serving Component

![TF-Chest Xray Pipeline](pictures/3-serving.PNG)

![TF-Network Traffic Pipeline](pictures/8-show-table.PNG)
