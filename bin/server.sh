#!/usr/bin/env bash
image=mece4520

if [ -z "$(docker image ls | grep ${image})" ]
then
	echo "The docker image ${image} is not found."
	echo "Have you run 'make build' yet?"
fi

docker run \
	--rm -it \
	-p 8888:8888 \
	-v "${PWD}":/home/jovyan/work \
	${image} \
	start-notebook.sh --NotebookApp.password='' --NotebookApp.token=''
