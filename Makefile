.PHONY: install build run

install:
	pip3 install -r requirements.txt
	cd ui && npm install && npm run build

build:
	cd ui && npm run build

run:
	python3 cli.py serve
