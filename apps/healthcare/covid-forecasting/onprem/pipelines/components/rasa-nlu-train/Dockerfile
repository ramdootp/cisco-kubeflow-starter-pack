FROM ubuntu:18.04

RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
     ca-certificates python3-dev python3-pip python3-setuptools python3-wheel wget \
	build-essential libssl-dev libffi-dev

RUN pip3 install rasa-nlu tornado spacy sklearn_crfsuite \ 
       && python3 -m spacy download en

ADD rasa/ /ml

ADD ./src/nlu_train.sh /bin/.
RUN chmod +x /bin/nlu_train.sh

ENTRYPOINT ["/bin/nlu_train.sh"]
