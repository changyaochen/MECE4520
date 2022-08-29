.PHONY: build serve

init:
	@chmod +x bin/setup.sh bin/build.sh bin/serve.sh

setup: init
	@bin/setup.sh

build: setup
	@bin/build.sh

serve: setup
	@bin/serve.sh
