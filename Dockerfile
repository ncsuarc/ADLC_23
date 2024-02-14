# If not using CUDA: swap this for ubuntu:22.04
FROM nvcr.io/nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04

ARG PORT=8087
ARG OPENCV_VERSION=4.5.4
# Prevent interactions from tcl?
ENV DEBIAN_FRONTEND noninteractive


# Install python
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    python3 \
	python3-pip 

RUN python3 -m pip install --upgrade pip

## Tesseract
RUN apt install -y \
    tesseract-ocr \
     libtesseract-dev \
     libleptonica-dev \
     pkg-config \
     curl

# Tessdata (english and OSD)
RUN mkdir /tessdata
RUN curl -LJ -o /tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
RUN curl -LJ -o /tessdata/osd.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/osd.traineddata

# ## OpenCV 
RUN apt-get install -y \
    libopencv-dev \
    python3-opencv

WORKDIR /adlc

## pip requirements
COPY requirements.txt ./

RUN cat requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements.txt

# Copy source folder
COPY ./adlc ./adlc
COPY ./data ./data
COPY ./img ./img
COPY ./test ./test

CMD uvicorn adlc.main:app --host 0.0.0.0 --port $PORT