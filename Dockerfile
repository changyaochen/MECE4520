FROM jupyter/datascience-notebook:ubuntu-22.04

USER root
RUN apt-get update && apt-get -y install graphviz

WORKDIR /home/jovyan

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
