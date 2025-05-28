
# Use the official NVIDIA CUDA base image with Ubuntu 20.04 and CUDA 11.1
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

EXPOSE 5000

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt install software-properties-common -y

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y \
    wget \
    gcc-9 \
    g++-9 \
    python3.12 \
    python3.12-dev \
    ca-certificates \
    curl \
    git \
    gfortran \
    && apt-get clean

RUN apt-get install ffmpeg libsm6 libxext6 libnss3 libegl1  -y

RUN ln -sf /usr/bin/python3.12 /usr/bin/python

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda create -y -n myenv python=3.12

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

WORKDIR /app

ADD ./requirements.txt .

RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_trf

ADD . .

CMD ["conda", "run", "-n", "myenv", "/bin/bash", "-c", "cd /app && python src/build_adj_embeddings.py"]