# COVID-19 Dataset on UCS infrastructure

COVID-19 ( Novel Corona Virus) has been declared as a global health emergency by WHO,
as it has been taking it's toll in many of the countries across the globe.

<img src="./pictures/corona_virus.jpg" width="500" align="middle"/>

## COVID-19 Forecasting

* This app focuses on forecasting/predicting the number of new cases & the number of new
fatalities that may occur in specific country-regions of the globe on specific forthcoming dates,
given the number of cases & fatalities that have already occurred in the foregone dates.

* The dataset involved here is publicly available at [COVID Train & Test Data](https://www.kaggle.com/c/covid19-global-forecasting-week-4/data).

* Model Training is implemented using Keras & LSTM ( Long Short Term Memory) architecture.

     * LSTM are a special kind of RNN(Recurrent Neural Network), capable of learning long-term dependencies.

     * LSTMs have internal mechanisms called gates that can learn which data in a sequence is important to
keep or throw away. By doing that, it can pass relevant information down the long chain of sequences
to make predictions.

There are multiple ways of training & deploying COVID-19 forecasting model on Kubeflow for this application:
  - [Using Kubeflow Pipelines](./pipelines)
  - [Using Jupyter notebook server from Kubeflow](./notebook)
  - [Using Kubeflow Fairing](./fairing)

## COVID-19 FAQ Bot

* This app is designed as a bot to answer the FAQs that are put forth related to COVID-19, also populating the information from various leading health sources/organisations possessing such information.

For FAQ Bot implemented as a Kubeflow pipeline, please refer [COVID-19 FAQ Bot](./pipelines/faq-bot).

## Chest X-ray diagnosis for COVID-19

* This app develops a chest X-ray model to understand and identify the covid infection from X-ray images.
* This would give physicians an edge and allow them to act with more confidence while they wait for the analysis of a radiologist by having a digital second opinion to confirm their assessment of a patient's condition. 

For Chest X-ray diagnosis implemented as a Kubeflow pipeline, please refer [Chest X-ray diagnosis](./pipelines/chest-xray).