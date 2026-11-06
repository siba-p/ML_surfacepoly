.PHONY: install install-dev data train-forward train-backward train-extended predict test lint clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,notebooks]"

data:            ## rebuild data/splits from data/raw
	python -m mlsurfpoly.cli.prepare_data

train-forward:
	python -m mlsurfpoly.cli.train_forward --model hybrid --epochs 2000

train-backward:
	python -m mlsurfpoly.cli.train_backward \
	  --forward-checkpoint models/pretrained/forward/HybridCNN/canonical_forward_model.keras \
	  --train-seq data/splits/fdX_train.npy --train-pmf data/splits/fdY_train.npy \
	  --valid-seq data/splits/fdX_valid.npy --valid-pmf data/splits/fdY_valid.npy

train-extended:
	python -m mlsurfpoly.cli.train_extended \
	  --forward-checkpoint models/pretrained/forward/HybridCNN/canonical_forward_model.keras \
	  --sequence-train data/splits/fdX_train.npy --pmf-train data/splits/fdY_train.npy --extra-train data/splits/bxX_train.npy \
	  --sequence-valid data/splits/fdX_valid.npy --pmf-valid data/splits/fdY_valid.npy --extra-valid data/splits/bxX_valid.npy

predict:
	python -m mlsurfpoly.cli.predict_forward \
	  --checkpoint models/pretrained/forward/HybridCNN/canonical_forward_model.keras \
	  --inputs data/splits/fdX_test.npy

test:
	pytest

lint:
	ruff check mlsurfpoly tests

clean:
	rm -rf models/runs .pytest_cache build dist *.egg-info
