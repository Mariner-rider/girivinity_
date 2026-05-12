.PHONY: train-tokeniser crawl-domains build-dataset train-model finetune-domains quantise run verify-3b

train-tokeniser:
	python -m model.tokeniser

crawl-domains:
	python -m model.domain_trainer --crawl --domain all

build-dataset:
	python -m model.domain_trainer --mix

train-model:
	python -m model.train \
		--data data/domain_training/mixed.jsonl \
		--tokeniser models/tokeniser/tokeniser.json \
		--output models/base

finetune-domains:
	python -m model.domain_trainer --finetune --dataset data/domain_training/mixed.jsonl

quantise:
	python -m model.quantise --weights models/base/final --output models/girivinity_quantised --quant Q4_K_M

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000


verify-3b:
	pytest tests/test_architecture_3b.py -v
