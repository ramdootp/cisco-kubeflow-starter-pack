FROM tensorflow/tensorflow:1.7.0-py3
RUN pip install kubeflow-katib kubernetes==10.0.1 dill
ADD chest-xray-katib.py  /opt/chest-xray-katib.py
RUN chmod +x /opt/chest-xray-katib.py 
ENTRYPOINT ["/usr/bin/python"]
CMD ["/opt/chest-xray-katib.py"]

