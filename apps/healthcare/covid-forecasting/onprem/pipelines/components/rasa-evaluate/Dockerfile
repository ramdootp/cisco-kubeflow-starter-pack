FROM ubuntu:18.04

RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
     ca-certificates python3-dev python3-pip python3-setuptools python3-wheel wget \
	build-essential libssl-dev libffi-dev

RUN pip3 install rasa-core tornado spacy sklearn_crfsuite tensorflow==1.13.1 matplotlib \
           && python3 -m spacy download en

RUN DEBIAN_FRONTEND=noninteractive apt install python3-tk -y

ADD rasa/ /ml
COPY matplotlibrc /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/matplotlibrc

ADD ./src/e2e-eval.sh /bin/.
RUN chmod +x /bin/e2e-eval.sh

ENTRYPOINT ["/bin/e2e-eval.sh"]
