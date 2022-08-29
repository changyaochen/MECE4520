FROM jupyter/datascience-notebook:ubuntu-22.04

WORKDIR /home/jovyan

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
