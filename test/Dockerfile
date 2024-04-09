FROM python:3.10-slim

ENV DEBIAN_FRONTEND noninteractive

# RUN apt-get update &&\
#     apt-get install -y exempi

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install --upgrade Pillow requests defusedxml
    # python-xmp-toolkit 

WORKDIR /adlc

COPY . ./test

CMD python3 ./test/test_adlc.py