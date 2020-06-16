# Chest-Xray Predicton using pipelines

## What we're going to build

Train and serve chest-xray model using KF pipeline, and predict xray result from Jupyter notebook.

![TF-Chest Xray Pipeline](pictures/0-xray-graph.PNG)

## Infrastructure Used

* Cisco UCS - C240M5 and C480ML

## Setup

### Install NFS server (if not installed)

To install NFS server follow [steps](../../../../../networking/ble-localization/onprem/pipelines#install-nfs-server-if-not-installed)

### Create Jupyter Notebook Server

Follow the [steps](./../../notebook#create--connect-to-jupyter-notebook-server)  to create Jupyter Notebook in Kubeflow

### Notebook to generate dataset and store it in kubeflow minio

Upload dataset-builder-minio-store.ipynb file from [here](./dataset-builder-minio-store.ipynb) and run all the cells to download and generate image dataset and store it in kubeflow minio.

This step should run only once

### Upload Notebook file for pipeline

Upload chest-xray-pipeline-deployment.ipynb file from [here](./chest-xray-pipeline-deployment.ipynb)

### Run Chest Xray Pipeline

Open the chest-xray-pipeline-deployment.ipynb file and run pipeline

Clone git repo

![TF-Chest Xray  Pipeline](pictures/1-git-clone.png)

Loading Components

![TF-BLERSSI Pipeline](pictures/2-load-compoents.PNG)

Run Pipeline

![TF-Chest Xray Pipeline](pictures/2-run-pipeline.PNG)

Once chest-xray pipeline is executed Experiment and Run link will generate and displayed as output

![TF-Chest Xray Pipeline](pictures/3-exp-link.PNG)

Click on latest experiment which is created

![TF-Chest Xray Pipeline](pictures/4-pipeline-created.PNG)

Pipeline components execution can be viewed as below
Logs of chest-xray training component
![TF-Chest Xray Pipeline](pictures/6-pipeline-completed.PNG)

Logs of serving component

![TF-Chest Xray Pipeline](pictures/3-serving.PNG)

![TF-Network Traffic Pipeline](pictures/8-show-table.PNG)
