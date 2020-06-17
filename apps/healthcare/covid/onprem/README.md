# COVID-19 models on UCS infrastructure

<!-- vscode-markdown-toc -->
* [COVID-19](#covid19)
     * [Introduction](#Introduction)
     * [COVID-19 Forecasting](#COVIDForecasting)
       * [Dataset](#Dataset)
       * [Model](#Model)
       * [Implementation](#ForecastImplementation)
     * [COVID-19 FAQ Bot](#COVIDFaqBot)
       * [Implementation](#FAQImplementation)
     * [Chest X-ray Diagnosis](#chestXrayDiagnosis)
       * [Implementation](#DiagnosisImplementation) 

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# <a name='COVID19'></a>**COVID-19**

## <a name='Introduction'></a>Introduction
COVID-19 ( Novel Corona Virus) has been declared as a global health emergency by WHO,
as it has been taking it's toll in many of the countries across the globe.

<img src="./pictures/corona_virus.jpg" width="500" align="middle"/>

## <a name='COVIDForecasting'></a>COVID-19 Forecasting

This app focuses on forecasting/predicting the number of new cases & the number of new
fatalities that may occur in specific country-regions of the globe on specific forthcoming dates,
given the number of cases & fatalities that have already occurred in the foregone dates.

### <a name='Dataset'></a>*Dataset*
The dataset involved here is publicly available at [COVID Train & Test Data](https://www.kaggle.com/c/covid19-global-forecasting-week-4/data).


### <a name='Model'></a>*Model*

 Model Training is implemented using Keras & LSTM ( Long Short Term Memory) architecture.

     * LSTM are a special kind of RNN(Recurrent Neural Network), capable of learning long-term dependencies.

     * LSTMs have internal mechanisms called gates that can learn which data in a sequence is important to keep or throw away. By doing that, it can pass relevant information down the long chain of sequences to make predictions.

### <a name='ForecastImplementation'></a>*Implementation*

There are multiple ways of training & deploying COVID-19 forecasting model on Kubeflow for this application:
  - [Using Kubeflow Pipelines](./pipelines)
  - [Using Jupyter notebook server from Kubeflow](./notebook)
  - [Using Kubeflow Fairing](./fairing)

## <a name='COVIDFaqBot'></a>COVID-19 FAQ Bot

 This app is designed as a bot to answer the FAQs that are put forth related to COVID-19, also populating the information from various leading health sources/organisations possessing such information.

### <a name='FAQImplementation'></a>*Implementation*

For FAQ Bot implemented as a Kubeflow pipeline, please refer [COVID-19 FAQ Bot](./pipelines/faq-bot).

## <a name='chestXrayDiagnosis'></a>Chest X-ray diagnosis for COVID-19

  This app develops a chest X-ray model to understand and identify the covid infection from X-ray images.

  This would give physicians an edge and allow them to act with more confidence while they wait for the analysis of a radiologist by having a digital second opinion to confirm their assessment of a patient's condition.

### <a name='DiagnosisImplementation'></a>*Implementation*   

For Chest X-ray diagnosis implemented as a Kubeflow pipeline, please refer [Chest X-ray diagnosis](./pipelines/chest-xray).
