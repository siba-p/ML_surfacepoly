all: preprocess train evaluate

preprocess:
	python scripts/data_prep.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

