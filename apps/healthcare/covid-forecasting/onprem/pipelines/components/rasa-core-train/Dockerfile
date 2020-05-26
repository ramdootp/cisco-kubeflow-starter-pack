FROM ubuntu:18.04

RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
     ca-certificates python3-dev python3-pip python3-setuptools python3-wheel wget \
	build-essential libssl-dev libffi-dev

RUN pip3 install rasa-core tornado

ADD rasa/ /ml

ADD ./src/core_train.sh /bin/.
RUN chmod +x /bin/core_train.sh

ENTRYPOINT ["/bin/core_train.sh"]
