# COVID-19 Forecasting on UCS infrastructure


COVID-19 ( Novel Corona Virus) has been declared as a global health emergency by WHO, 
as it has been taking it's toll in many of the countries across the globe.  

This app focuses on forecasting/predicting the number of new cases & the number of new 
fatalities that may occur in specific country-regions of the globe on specific forthcoming dates,
given the number of cases & fatalities that have already occurred in the foregone dates.

The dataset which we involve here is publicly available at [COVID Train & Test Data](https://www.kaggle.com/c/covid19-global-forecasting-week-4/data).


<img src="./pictures/corona_virus.jpg" width="500" align="middle"/>

Model Training is implemented using Keras & LSTM ( Long Short Term Memory) architecture.

LSTM are a special kind of RNN(Recurrent Neural Network), capable of learning long-term dependencies.
LSTMs have internal mechanisms called gates that can learn which data in a sequence is important to 
keep or throw away. By doing that, it can pass relevant information down the long chain of sequences 
to make predictions.


There are multiple ways of training & deploying the model in Kubeflow for this application:
  - [Using Kubeflow Pipelines](./pipelines)
  - [Using Jupyter notebook server from Kubeflow](./notebook)
  - [Using Kubeflow Fairing](./fairing)
