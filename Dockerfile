FROM jupyter/datascience-notebook:python-3.10.11

USER root
RUN apt-get update && apt-get -y install graphviz

WORKDIR /home/jovyan

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
