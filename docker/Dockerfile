FROM tensorflow/tensorflow:latest-gpu

#Install dependencies
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y git \
    vim 

WORKDIR /home

