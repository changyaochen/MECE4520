.PHONY: setup build server

setup:
	@bin/setup.sh

build: setup
	@bin/build.sh

server: setup
	@bin/server.sh
