FROM python:3.8

RUN pip install --upgrade pip

RUN apt-get update && yes | apt-get upgrade

RUN pip install tensorflow==2.7.0

RUN apt-get install -y git python3-pip

RUN apt-get install -y protobuf-compiler python3-pil python3-lxml

RUN mkdir -p /tensorflow/models

RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

RUN apt-get update && apt-get install -y cmake

RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /tensorflow/models/research

RUN protoc object_detection/protos/*.proto --python_out=.

RUN export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim

WORKDIR /app

COPY app/ .

COPY req.txt .

RUN pip install --default-timeout=100 -r req.txt

WORKDIR /

COPY tuned_model/ .

COPY initializer.sh .

RUN chmod 777 initializer.sh

EXPOSE 8000

ENTRYPOINT ["./initializer.sh"]