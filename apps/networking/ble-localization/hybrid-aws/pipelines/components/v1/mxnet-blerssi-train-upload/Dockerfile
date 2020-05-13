FROM python:3.5
WORKDIR /opt/
COPY . /opt/
RUN pip install numpy pandas sklearn mxnet boto3 python-dotenv sagemaker six clize
CMD [ "python", "/opt/mx-dnn.py" ]
