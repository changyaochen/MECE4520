#!/usr/bin/env bash

if ! command -v docker &> /dev/null
then
	echo "Docker is not found."
	echo "Please install via: https://docs.docker.com/engine/install/"
	exit
fi

docker ps &> /dev/null
if [ $? -ne 0 ]
then
	echo "Is the Docker Engine running?"
	exit
fi
