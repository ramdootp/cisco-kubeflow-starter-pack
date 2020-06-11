FROM tensorflow/tensorflow:1.15.2-gpu
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev 
RUN apt-get install -y wget
RUN pip install -U scikit-learn
RUN pip install pandas opencv-python pydicom pillow imutils
ADD src/covid-xray-model.py  /opt/covid-xray-model.py
ADD src/data/datasets /opt/data/datasets
RUN chmod +x /opt/covid-xray-model.py  /opt/data/datasets
WORKDIR /opt/
CMD python covid-xray-model.py
