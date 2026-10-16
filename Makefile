.PHONY: install data train test run

install:
	pip install -r requirements.txt

data:
	python scripts/run_pipeline.py --data-only

train:
	python scripts/run_pipeline.py

test:
	python -m unittest discover -s tests -v

run: train
